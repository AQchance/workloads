"""
ICONQ → Bi-LSTM on 258-query batch traces (K4 + K8).
97-dim interaction vector, absolute runtime prediction.
"""
import os, re, csv, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DIR = '/home/anqian/Desktop/my_lab/workloads'
RESOURCE_PATH = '/home/anqian/code/python/workloads/collect_concurrent/tabpfn_258_predictions_oof.json'
# XGBoost cache for ICONQ runtime feature
XGB_PATH = os.path.join(DIR, 'checkpoints/oof_xgboost_k5.json')

TRACES = [
    '/home/anqian/code/python/workloads/final_queries/k4_batch_trace.csv',
    '/home/anqian/code/python/workloads/final_queries/k8_batch_trace.csv',
]
OUT_DIR = '/home/anqian/code/python/workloads/final_queries'

SEED = 42
EPOCHS = 250

# ═══════════ ICONQ Features ═══════════

OP_TYPES = [
    'TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
    'IndexLookUp', 'IndexReader',
    'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
    'HashAgg', 'StreamAgg',
    'Sort', 'TopN', 'Window',
    'ExchangeSender', 'ExchangeReceiver',
    'Projection', 'Selection',
]
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer',
          'nation', 'region']


def extract_iconq_query_features(qid, xgb_cache):
    pf = os.path.join(DIR, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        return None
    with open(pf) as f:
        plan = f.read()
    oc = {op: 0 for op in OP_TYPES}
    oe = {op: 0.0 for op in OP_TYPES}
    tc = {t: 0.0 for t in TABLES}
    for line in plan.split('\n'):
        if '\t' not in line or line.startswith('--'):
            continue
        parts = line.lstrip(' │├└─').split('\t')
        if len(parts) < 5: continue
        raw_op = re.sub(r'^[│├└─\s]+', '', parts[0].strip())
        op = re.sub(r'\(Build\)|\(Probe\)', '', raw_op).strip()
        op = re.sub(r'_\d+$', '', op)
        try: est = float(parts[1].strip())
        except ValueError: est = 1.0
        if op in oc: oc[op] += 1; oe[op] += est
        oi = parts[4].strip() if len(parts) > 4 else ''
        for t in TABLES:
            if t in oi.lower(): tc[t] = max(tc[t], est)
    feat = []
    for op in OP_TYPES:
        feat.append(float(oc[op])); feat.append(math.log(1 + oe[op]))
    for t in TABLES:
        feat.append(math.log(1 + tc[t]))
    lat_entry = xgb_cache.get(qid, {})
    feat.append(lat_entry.get('lat', math.log(1 + 10.0)))
    return np.array(feat, dtype=np.float32)


def build_interaction_vector(qi_feat, qj_feat, ti, tj):
    return np.concatenate([
        qi_feat, qj_feat,
        np.array([abs(ti - tj), 1.0 if ti < tj else 0.0,
                  1.0 if tj < ti else 0.0], dtype=np.float32),
    ])


# ═══════════ Build Sequences ═══════════

def build_sequences(trace_csv, feat_cache):
    timeline = []
    with open(trace_csv) as f:
        for row in csv.DictReader(f):
            t0 = float(row['start'])
            rt = float(row['runtime']); a = 60.0 if row['status'] == 'penalty' else rt
            timeline.append((t0, t0 + a, row['qid'], a, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))

    X, y_abs = [], []
    for qi, si, ei, rti, sti, ov in ci:
        if sti == 'penalty' or qi not in feat_cache:
            continue
        qj_feat = feat_cache[qi]
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov if oq in qid_info and oq in feat_cache]
        peers.sort()
        for osv, oq in peers:
            seq.append(build_interaction_vector(feat_cache[oq], qj_feat, osv, si))
        if seq: X.append(seq); y_abs.append(rti)
    return X, y_abs


# ═══════════ Model ═══════════

class ConcurrentDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X); self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]


def collate_fn(batch):
    X, l, y = zip(*batch)
    si = torch.argsort(torch.stack(l), descending=True)
    return (torch.stack([X[i] for i in si]), torch.stack([l[i] for i in si]),
            torch.stack([y[i] for i in si]))


class IconqBiLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim=128, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Sequential(nn.Linear(input_dim, embed_dim),
                                        nn.Linear(embed_dim, embed_dim))
        self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                              dropout=dropout, batch_first=True, bidirectional=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size // 2),
                                           nn.Linear(hidden_size // 2, 1))
        for m in [self.embedding, self.bilstm, self.output_layer]:
            for n, p in m.named_parameters():
                if 'weight' in n: nn.init.xavier_uniform_(p.data)
                elif 'bias' in n: nn.init.constant_(p.data, 0.0)

    def forward(self, X, lengths):
        if X.shape[1] > 1:
            X = torch.transpose(X, 1, 2); X = self.bn(X); X = torch.transpose(X, 1, 2)
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[torch.arange(len(lengths)), lengths - 1]
        return self.output_layer(output).squeeze(-1)


# ═══════════ Training ═══════════

def pad_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    if Xm is None:
        mask = np.zeros_like(Xa)
        for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
        Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        diff = ((Xa - Xm) * mask) ** 2; Xs = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    return (Xa - Xm) / Xs, lens, (yl - ym) / ys, Xm, Xs, ym, ys


def train(train_ds, val_ds, test_ds, ym, ys, input_dim, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = IconqBiLSTM(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            model.eval(); ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader: ap.append(model(X.to(device), l).cpu().numpy()); at.append(y.numpy())
            pz, tz = np.concatenate(ap), np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01); tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val: best_val = med; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  [{label}] E{epoch:3d} val={med:.2f}x best={best_val:.2f}x  ({n_params:,} params)')

    model.load_state_dict(best_state); model.eval(); ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader: ap.append(model(X.to(device), l).cpu().numpy()); at.append(y.numpy())
    pz, tz = np.concatenate(ap), np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01); tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    return qe, best_state


# ═══════════ Main ═══════════

def main():
    print('=' * 55)
    print('ICONQ → Bi-LSTM on 258-query K4+K8 Traces')
    print('=' * 55)

    with open(XGB_PATH) as f: xgb_cache = json.load(f)
    print(f'[1] XGBoost latency cache: {len(xgb_cache)} queries')

    # Extract ICONQ features for all trace queries
    all_qids = set()
    for trace_csv in TRACES:
        with open(trace_csv) as f:
            for row in csv.DictReader(f): all_qids.add(row['qid'])

    feat_cache = {}
    for qid in all_qids:
        f = extract_iconq_query_features(qid, xgb_cache)
        if f is not None: feat_cache[qid] = f
    print(f'[2] ICONQ features: {len(feat_cache)} queries, dim={len(list(feat_cache.values())[0])}')

    # Collect all qids
    unique_qids = sorted(set(
        q for trace_csv in TRACES for q in
        [r['qid'] for r in csv.DictReader(open(trace_csv))] if q in feat_cache))
    print(f'[3] Unique queries: {len(unique_qids)}')

    np.random.seed(SEED); np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8); n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr]); val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'    Split: {len(train_qids)}/{len(val_qids)}/{len(test_qids)}')

    X_tr, X_va, X_te = [], [], []; y_tr, y_va, y_te = [], [], []
    for trace_csv in TRACES:
        timeline = []
        with open(trace_csv) as f:
            for row in csv.DictReader(f):
                t0 = float(row['start']); rt = float(row['runtime'])
                a = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((t0, t0 + a, row['qid'], a, row['status']))
        qid_info = {q: (s, e) for s, e, q, _, _ in timeline}
        ci = []
        for i, (si, ei, qi, rti, sti) in enumerate(timeline):
            ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si]
            ci.append((qi, si, ei, rti, sti, ov))
        for qi, si, ei, rti, sti, ov in ci:
            if sti == 'penalty' or qi not in feat_cache: continue
            qj_feat = feat_cache[qi]; seq = []
            peers = [(qid_info[oq][0], oq) for oq in ov if oq in qid_info and oq in feat_cache]
            peers.sort()
            for osv, oq in peers:
                seq.append(build_interaction_vector(feat_cache[oq], qj_feat, osv, si))
            if not seq: continue
            if qi in train_qids: X_tr.append(seq); y_tr.append(rti)
            elif qi in val_qids: X_va.append(seq); y_va.append(rti)
            else: X_te.append(seq); y_te.append(rti)

    print(f'[4] Seqs: {len(X_tr)}/{len(X_va)}/{len(X_te)}')
    d_in = len(X_tr[0][0])
    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va), max(len(s) for s in X_te))
    print(f'    Dim={d_in}, max_len={ml}')

    Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_normalize(X_tr, y_tr, ml)
    Xn_va, l_va, yn_va, _, _, _, _ = pad_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
    Xn_te, l_te, yn_te, _, _, _, _ = pad_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

    print(f'[5] Training...')
    qe, best_state = train(ConcurrentDataset(Xn_tr, l_tr, yn_tr),
                            ConcurrentDataset(Xn_va, l_va, yn_va),
                            ConcurrentDataset(Xn_te, l_te, yn_te), ym, ys, d_in, 'ICONQ')

    torch.save(best_state, os.path.join(OUT_DIR, 'bilstm_iconq.pt'))
    np.savez(os.path.join(OUT_DIR, 'bilstm_iconq_norm.npz'),
             X_mean=Xm, X_std=Xs, y_mean=ym, y_std=ys)

    n = len(qe)
    print(f'\n{"="*55}')
    print(f'ICONQ → BiLSTM (K4+K8, 258 queries):')
    for pct, label in [(10, 'P10'), (50, 'P50'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
        print(f'  {label}: {qe[min(int(n*pct/100), n-1)]:.2f}x')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
