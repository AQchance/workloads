"""
TabPFN → Bi-LSTM concurrent slowdown prediction.
Stage 1: TabPFN OOF 5-dim resource predictions (pre-cached)
Stage 2: Bi-LSTM with 17-dim ICONQ-style interaction features
Hard split: 80/10/10 query-level, seed=42
"""
import os, sys, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']
SEED = 42
EPOCHS = 250


# ═══════════ Stage 1: Load TabPFN OOF Predictions ═══════════

def load_resource_cache():
    path = os.path.join(CKPT_DIR, 'oof_tabpfn_k5.json')
    with open(path) as f:
        return json.load(f)


# ═══════════ Stage 2: Build Interaction Sequences ═══════════

def resource_conflict(t, c):
    t, c = np.array(t), np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_sequences(data_list, cache, qid_info):
    """17-dim ICONQ-style interaction features:
       target(5) + peer(5) + timing(2) + conflict(5) = 17"""
    X, y_ratio = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if sti == 'penalty' or qi not in cache:
            continue
        pred_i = cache[qi]
        serial_lat = max(math.exp(pred_i['lat']) - 1, 0.5)
        qv = [pred_i[d] for d in DIMS]
        tr_ = [pred_i[d] for d in DIMS]
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov
                 if oq in qid_info and oq in cache]
        peers.sort()
        for osv, oq in peers:
            pred_j = cache[oq]
            ovv = [pred_j[d] for d in DIMS]
            oc = [pred_j[d] for d in DIMS]
            c = resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    return X, y_ratio


# ═══════════ Stage 3: Bi-LSTM ═══════════

class RatioDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]


def collate_fn(batch):
    X, l, y = zip(*batch)
    si = torch.argsort(torch.stack(l), descending=True)
    return (torch.stack([X[i] for i in si]),
            torch.stack([l[i] for i in si]),
            torch.stack([y[i] for i in si]))


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


# ═══════════ Training ═══════════

def pad_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq):
        Xa[i, :len(s)] = s
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    if Xm is None:
        mask = np.zeros_like(Xa)
        for i, s in enumerate(X_seq):
            mask[i, :len(s)] = 1.0
        Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        diff = ((Xa - Xm) * mask) ** 2
        Xs = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    return (Xa - Xm) / Xs, lens, (yl - ym) / ys, Xm, Xs, ym, ys


def train(train_ds, val_ds, test_ds, ym, ys, input_dim):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = BiLSTM(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    ap.append(model(X.to(device), l).cpu().numpy())
                    at.append(y.numpy())
            pz, tz = np.concatenate(ap), np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  E{epoch:3d} val={med:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state)
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader:
            ap.append(model(X.to(device), l).cpu().numpy())
            at.append(y.numpy())
    pz, tz = np.concatenate(ap), np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
    tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    return qe


# ═══════════ Main ═══════════

def main():
    print('=' * 55)
    print('TabPFN → Bi-LSTM (17-dim, Hard Split)')
    print('=' * 55)

    cache = load_resource_cache()
    print(f'[1] TabPFN OOF cache: {len(cache)} queries')

    # Load timeline
    traces = ['collect_concurrent/trace_2_mixed.csv',
              'collect_concurrent/trace_3_fixed_mixed.csv',
              'collect_concurrent/trace_4_fixed_mixed.csv']
    timeline = []
    for tf in traces:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime'])
                actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']),
                                 float(row['start']) + actual,
                                 row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    # Load known query set (from timeline)
    with open(os.path.join(ROOT, 'lstm/timeline_qids.txt')) as f:
        valid_qids = set(line.strip() for line in f)
    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in valid_qids))
    print(f'[2] Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # 80/10/10 hard split
    np.random.seed(SEED)
    np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8)
    n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr])
    val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'[3] Split: {len(train_qids)} train / {len(val_qids)} val / {len(test_qids)} test')

    # Build event list
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]

    X_tr, y_tr = build_sequences(train_d, cache, qid_info)
    X_va, y_va = build_sequences(val_d, cache, qid_info)
    X_te, y_te = build_sequences(test_d, cache, qid_info)
    d_in = len(X_tr[0][0])
    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
             max(len(s) for s in X_te))
    print(f'[4] Seqs: {len(X_tr)}/{len(X_va)}/{len(X_te)}, max_len={ml}, dim={d_in}')

    Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_normalize(X_tr, y_tr, ml)
    Xn_va, l_va, yn_va, _, _, _, _ = pad_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
    Xn_te, l_te, yn_te, _, _, _, _ = pad_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

    print(f'[5] Training...')
    qe = train(RatioDataset(Xn_tr, l_tr, yn_tr),
               RatioDataset(Xn_va, l_va, yn_va),
               RatioDataset(Xn_te, l_te, yn_te), ym, ys, d_in)

    n = len(qe)
    print(f'\n{"="*55}')
    print(f'TabPFN → BiLSTM:')
    for pct, label in [(10, 'P10'), (50, 'P50'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
        print(f'  {label}: {qe[min(int(n*pct/100), n-1)]:.2f}x')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
