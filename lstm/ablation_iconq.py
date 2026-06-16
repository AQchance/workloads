"""
Systematic ICONQ ablation: test single-parameter changes independently.
Run: python lstm/ablation_iconq.py
"""

import os, sys, json, csv, math, re, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = '/home/anqian/Desktop/my_lab/workloads'
OUT = os.path.join(ROOT, 'lstm')
sys.path.insert(0, 'lstm')

OP19 = ['TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
        'IndexLookUp', 'IndexReader', 'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
        'HashAgg', 'StreamAgg', 'Sort', 'TopN', 'Window',
        'ExchangeSender', 'ExchangeReceiver', 'Projection', 'Selection']
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer', 'nation', 'region']

# Load GNN features
gnf = {}
for fn in ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
           'lstm/gnn_features_k4_fixed.json']:
    with open(os.path.join(ROOT, fn)) as f:
        gnf.update(json.load(f))


def mk_features():
    iconq = {}
    for qid in gnf:
        pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
        if not os.path.exists(pf): continue
        with open(pf) as f:
            plan = f.read()
        pred_lat = max(abs(float(gnf[qid]['gpu_resources'].get('lat', 1))), 0.5)
        oc = {o: 0 for o in OP19}
        oe = {o: 0.0 for o in OP19}
        tc = {t: 0.0 for t in TABLES}
        for line in plan.split('\n'):
            if '\t' not in line or line.startswith('--'): continue
            s = line.lstrip(' │├└─')
            parts = s.split('\t')
            if len(parts) < 5: continue
            rid = parts[0].strip()
            op = re.sub(r'^[│├└─\s]+', '', rid)
            op = re.sub(r'\(Build\)|\(Probe\)', '', op).strip()
            op = re.sub(r'_\d+$', '', op)
            try:
                est = float(parts[1].strip())
            except:
                est = 1.0
            if op in oc: oc[op] += 1; oe[op] += est
            oi = parts[4].strip() if len(parts) > 4 else ''
            for t in TABLES:
                if t in oi.lower(): tc[t] = max(tc[t], est)
        feat = [math.log(1 + pred_lat)]
        for o in OP19: feat.append(float(oc[o])); feat.append(math.log(1 + oe[o]))
        for t in TABLES: feat.append(math.log(1 + tc[t]))
        iconq[qid] = feat
    return iconq


# Build timeline and data
TRACES = ['collect_concurrent/trace_2_mixed.csv', 'collect_concurrent/trace_3_fixed_mixed.csv',
          'collect_concurrent/trace_4_fixed_mixed.csv']
timeline = []
for tf in TRACES:
    with open(os.path.join(ROOT, tf)) as f:
        for row in csv.DictReader(f):
            rt = float(row['runtime']); st = row['status']
            actual = 60.0 if st == 'penalty' else rt
            timeline.append((float(row['start']), float(row['start']) + actual,
                           row['qid'], actual, st))

qid_info = {}
for s, e, q, _, _ in timeline:
    qid_info[q] = (s, e)

ci = [(qi, si, ei, rti, sti,
       [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si])
      for i, (si, ei, qi, rti, sti) in enumerate(timeline)]

mt = max(e for _, _, e, _, _, _ in ci)
split_time = mt * 0.7
train_d = [c for c in ci if c[1] < split_time]
test_d = [c for c in ci if c[1] >= split_time]


def build(dl, iconq):
    X, ya = [], []
    for qi, si, ei, rti, sti, ov in dl:
        if qi not in iconq or sti == 'penalty': continue
        qv = iconq[qi]
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in iconq and oq in qid_info]
        if not oi: continue
        oi.sort()
        seq = []
        for osv, oq in oi:
            seq.append(qv + iconq[oq] + [si - osv, 1.0 if osv < si else 0.0])
        if seq: X.append(seq); ya.append(rti)
    return X, ya


def build_data(iconq):
    X_tr, y_tr = build(train_d, iconq)
    X_te, y_te = build(test_d, iconq)
    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te))
    d = len(X_tr[0][0])
    Xa = np.zeros((len(X_tr), ml, d), dtype=np.float32)
    for i, s in enumerate(X_tr): Xa[i, :len(s)] = s
    Xta = np.zeros((len(X_te), ml, d), dtype=np.float32)
    for i, s in enumerate(X_te): Xta[i, :len(s)] = s
    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_tr): mask[i, :len(s)] = 1.0
    Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    yl = np.log(1 + np.array(y_tr, dtype=np.float32))
    ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    return Xa, Xta, Xm, Xs, y_tr, y_te, ym, ys, ml, d, yl


class DS:
    def __init__(self, X, l, y, ym, ys):
        self.X = torch.FloatTensor(X); self.lengths = torch.LongTensor(l)
        self.y = torch.FloatTensor(y); self.y_mean = ym; self.y_std = ys

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]


def cf(batch):
    X, l, y = zip(*batch)
    si = torch.argsort(torch.stack(l), descending=True)
    return torch.stack([X[i] for i in si]), torch.stack([l[i] for i in si]), torch.stack(
        [y[i] for i in si])


def test_variant(name, model_factory, tr_loader, el_loader, ym, ys, d, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    m = model_factory(d).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=2e-5)
    bv, bs = float('inf'), None
    for epoch in range(1, 81):
        m.train()
        for X, l, y in tr_loader:
            X, y = X.cuda(), y.cuda(); opt.zero_grad()
            loss = nn.functional.huber_loss(m(X, l), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        m.eval(); ap, at = [], []
        with torch.no_grad():
            for X, l, y in el_loader:
                X, y = X.cuda(), y.cuda(); ap.append(m(X, l).cpu().numpy()); at.append(y.cpu().numpy())
        p = np.concatenate(ap); t = np.concatenate(at)
        pr = np.maximum(np.exp(p * ys + ym) - 1, 0.01); tr = np.maximum(np.exp(t * ys + ym) - 1, 0.01)
        med = np.median(np.maximum(pr / tr, tr / pr))
        if med < bv: bv = med; bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); ap, at = [], []
    with torch.no_grad():
        for X, l, y in el_loader:
            X, y = X.cuda(), y.cuda(); ap.append(m(X, l).cpu().numpy()); at.append(y.cpu().numpy())
    p = np.concatenate(ap); t = np.concatenate(at)
    pr = np.maximum(np.exp(p * ys + ym) - 1, 0.01); tr = np.maximum(np.exp(t * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr)); n = len(qe)
    r = (qe[n // 2], qe[int(n * 0.9)], qe[int(n * 0.95)])
    print(f'  {name:40s} P50={r[0]:.2f}x P90={r[1]:.2f}x')
    return r


# ─── Model factories ───
def make_model(idim, bidir=True, bn=True, hid=256, nl=2, do=0.1, xavier=True):
    """Generic ICONQ-style LSTM factory."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.has_bn = bn
            if bn:
                self.bn = nn.BatchNorm1d(idim)
            self.emb = nn.Sequential(nn.Linear(idim, 128), nn.Linear(128, 128))
            hdim = hid * 2 if bidir else hid
            self.lstm = nn.LSTM(128, hid, nl, dropout=do, batch_first=True, bidirectional=bidir)
            self.out = nn.Sequential(nn.Linear(hdim, hid // 2), nn.Linear(hid // 2, 1))
            if xavier:
                for mod in [self.emb, self.lstm, self.out]:
                    for n, p in mod.named_parameters():
                        if 'weight' in n: nn.init.xavier_uniform_(p.data)
                        elif 'bias' in n: nn.init.constant_(p.data, 0.0)

        def forward(self, x, le):
            if self.has_bn and x.shape[1] > 1:
                x_t = torch.transpose(x, 1, 2)
                x_t = self.bn(x_t)
                x = torch.transpose(x_t, 1, 2)
            x = self.emb(x)
            p = nn.utils.rnn.pack_padded_sequence(x, le.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.lstm(p)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            return self.out(o[torch.arange(len(le)), le - 1]).squeeze(-1)

    return M().cuda()


if __name__ == '__main__':
    iconq = mk_features()
    print(f'Features: {len(iconq)} queries, dim={len(list(iconq.values())[0])}')

    Xa0, Xta0, Xm0, Xs0, y_tr0, y_te0, ym0, ys0, ml0, d0, yl0 = build_data(iconq)
    tr0 = DataLoader(DS((Xa0 - Xm0) / Xs0, np.array([len(s) for s in build(train_d, iconq)[0]], dtype=np.int32),
                        (yl0 - ym0) / ys0, ym0, ys0), batch_size=128, shuffle=True, collate_fn=cf)
    el0 = DataLoader(DS((Xta0 - Xm0) / Xs0, np.array([len(s) for s in build(test_d, iconq)[0]], dtype=np.int32),
                        (np.log(1 + np.array(y_te0, dtype=np.float32)) - ym0) / ys0, ym0, ys0), batch_size=256,
                     shuffle=False, collate_fn=cf)

    print(f'Train: {len(y_tr0)}, Test: {len(y_te0)}\n')
    print(f'  {"Variant":40s}  {"P50":>6s}  {"P90":>6s}  {"P95":>6s}')
    print('  ' + '-' * 65)

    r0 = test_variant('1. Baseline BiLSTM (repo exact)', lambda d: make_model(d), tr0, el0, ym0, ys0, d0)
    r1 = test_variant('2. Unidirectional (default lstm)', lambda d: make_model(d, bidir=False), tr0, el0, ym0, ys0, d0)
    r2 = test_variant('3. No BatchNorm', lambda d: make_model(d, bn=False), tr0, el0, ym0, ys0, d0)
    r3 = test_variant('4. hidden_size 256->64', lambda d: make_model(d, hid=64), tr0, el0, ym0, ys0, d0)
    r4 = test_variant('5. num_layers 2->1', lambda d: make_model(d, nl=1), tr0, el0, ym0, ys0, d0)
    r5 = test_variant('6. Default init (no Xavier)', lambda d: make_model(d, xavier=False), tr0, el0, ym0, ys0, d0)

    # 7. Remove predicted latency
    iconq_nolat = {q: [0.0] + f[1:] for q, f in iconq.items()}
    Xa7, Xta7, Xm7, Xs7, y_tr7, y_te7, ym7, ys7, ml7, d7, yl7 = build_data(iconq_nolat)
    tr7 = DataLoader(DS((Xa7 - Xm7) / Xs7, np.array([len(s) for s in build(train_d, iconq_nolat)[0]], dtype=np.int32),
                        (yl7 - ym7) / ys7, ym7, ys7), batch_size=128, shuffle=True, collate_fn=cf)
    el7 = DataLoader(
        DS((Xta7 - Xm7) / Xs7, np.array([len(s) for s in build(test_d, iconq_nolat)[0]], dtype=np.int32),
           (np.log(1 + np.array(y_te7, dtype=np.float32)) - ym7) / ys7, ym7, ys7), batch_size=256, shuffle=False,
        collate_fn=cf)
    r6 = test_variant('7. Remove predicted latency', lambda d: make_model(d), tr7, el7, ym7, ys7, d7)

    print('\n' + '=' * 65)
    print(f'  Baseline P50 = {r0[0]:.2f}x')
    for name, r in [('Unidirectional', r1), ('No BN', r2), ('Small hid', r3), ('1 Layer', r4),
                    ('Default init', r5), ('No pred lat', r6)]:
        delta = (r[0] - r0[0]) / r0[0] * 100
        print(f'  {name:25s}: P50={r[0]:.2f}x ({delta:+.0f}%)')
