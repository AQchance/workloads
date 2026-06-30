"""
Stage 2 only: Bi-LSTM with 17-dim ICONQ features.
Loads precomputed K-Fold OOF resource caches (zero leak).
"""
import os, sys, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT = os.path.join(ROOT, 'checkpoints')

# ═══ 17-dim ICONQ features ═══
# Target:  mem, disk, net, lat, cpures               = 5
# Peer:    mem, disk, net, lat, cpures               = 5
# Timing:  start_diff, is_before                     = 2
# Conflict: min(t,c)/(|t|+|c|) × 5 resources         = 5
#                                             Total  = 17

DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_sequences(data_list, cache, qid_info):
    X, y_ratio = [], []
    n_miss = 0
    for qi, si, ei, rti, sti, ov in data_list:
        if sti == 'penalty': continue
        pred_i = cache.get(qi)
        if pred_i is None: n_miss += 1; continue
        serial_lat = max(math.exp(pred_i['lat']) - 1, 0.5)
        qv = [pred_i[d] for d in DIMS]                    # 5
        tr_ = [pred_i[d] for d in DIMS]                   # 5
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov
                 if oq in qid_info and cache.get(oq) is not None]
        peers.sort()
        for osv, oq in peers:
            pred_j = cache[oq]
            ovv = [pred_j[d] for d in DIMS]               # 5
            oc = [pred_j[d] for d in DIMS]                # 5
            c = resource_conflict(tr_, oc)                 # 5
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    if n_miss: print(f'  Skipped {n_miss} queries')
    return X, y_ratio


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
    return torch.stack([X[i] for i in si]), torch.stack([l[i] for i in si]), \
           torch.stack([y[i] for i in si])


class BiLSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
    def forward(self, X, lengths):
        x = self.emb(X)
        p = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                               enforce_sorted=True)
        _, (hn, _) = self.lstm(p)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


def pad_and_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    if Xm is None:
        mask = np.zeros_like(Xa)
        for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
        Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        Xs = np.sqrt(((Xa - Xm)**2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    X_norm = (Xa - Xm) / Xs; y_norm = (yl - ym) / ys
    return X_norm, lens, y_norm, Xm, Xs, ym, ys


def train_bilstm(train_ds, val_ds, test_ds, ym, ys, input_dim, label, seed=42, epochs=250):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = BiLSTM(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            model.eval(); ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    X = X.to(device)
                    ap.append(model(X, l).cpu().numpy())
                    at.append(y.numpy())
            pz = np.concatenate(ap); tz = np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  [{label}] E{epoch:3d} val={med:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state); model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader:
            X = X.to(device)
            ap.append(model(X, l).cpu().numpy())
            at.append(y.numpy())
    pz = np.concatenate(ap); tz = np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
    tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    return qe, n_params


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    args = parser.parse_args()

    print('=' * 60)
    print('Bi-LSTM Stage 2 Only (17-dim ICONQ features)')
    print(f'Using precomputed K-Fold OOF caches (zero leak)')
    print(f'Seed={args.seed}, Epochs={args.epochs}')
    print('=' * 60)

    # Load timeline
    TRACES = ['collect_concurrent/trace_2_mixed.csv',
              'collect_concurrent/trace_3_fixed_mixed.csv',
              'collect_concurrent/trace_4_fixed_mixed.csv']
    timeline = []
    for tf in TRACES:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime'])
                actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']),
                                 float(row['start']) + actual,
                                 row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    # Load GNN features (known query set)
    gnf = {}
    for fn in ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
               'lstm/gnn_features_k4_fixed.json']:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))

    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))
    print(f'Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # 80/10/10 hard split
    np.random.seed(args.seed)
    np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8)
    n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr])
    val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'Split: {len(train_qids)} train / {len(val_qids)} val / {len(test_qids)} test')

    # Build concurrent event list
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]

    all_results = {}

    for model_name in ['TabPFN', 'XGBoost']:
        cache_path = os.path.join(CKPT, f'oof_{model_name.lower()}_k5.json')
        with open(cache_path) as f:
            cache = json.load(f)
        print(f'\n--- {model_name} → Bi-LSTM ---')
        print(f'  Loaded OOF cache: {len(cache)} queries')

        X_tr, y_tr = build_sequences(train_d, cache, qid_info)
        X_va, y_va = build_sequences(val_d, cache, qid_info)
        X_te, y_te = build_sequences(test_d, cache, qid_info)
        d_in = len(X_tr[0][0])
        ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
                 max(len(s) for s in X_te))
        print(f'  Seqs: {len(X_tr)}/{len(X_va)}/{len(X_te)}, max_len={ml}, dim={d_in}')

        Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_and_normalize(X_tr, y_tr, ml)
        Xn_va, l_va, yn_va, _, _, _, _ = pad_and_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
        Xn_te, l_te, yn_te, _, _, _, _ = pad_and_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

        qe, n_params = train_bilstm(
            RatioDataset(Xn_tr, l_tr, yn_tr),
            RatioDataset(Xn_va, l_va, yn_va),
            RatioDataset(Xn_te, l_te, yn_te),
            ym, ys, d_in, model_name, seed=args.seed, epochs=args.epochs)

        n = len(qe)
        all_results[model_name] = {
            'P10': qe[int(n*0.1)], 'P50': qe[n//2], 'P90': qe[int(n*0.9)],
            'P95': qe[int(n*0.95)], 'P99': qe[int(n*0.99)]
        }

    print('\n' + '=' * 60)
    print('FINAL: 17-dim Bi-LSTM with K-Fold OOF Resource Predictions')
    print(f'Zero leak, 80/10/10 split, Seed={args.seed}')
    print('=' * 60)
    print(f'\n  {"Method":<22} {"P10":>7} {"P50":>7} {"P90":>7} {"P95":>7} {"P99":>7}')
    print('  ' + '-' * 55)
    for name in ['TabPFN', 'XGBoost']:
        r = all_results[name]
        print(f'  {name+"->BiLSTM":<22} {r["P10"]:.2f}x {r["P50"]:.2f}x '
              f'{r["P90"]:.2f}x {r["P95"]:.2f}x {r["P99"]:.2f}x')
    t = all_results['TabPFN']; x = all_results['XGBoost']
    for k in ['P10', 'P50', 'P90', 'P95', 'P99']:
        print(f'  Δ{k}={(t[k]-x[k])/x[k]*100:+.1f}%', end='')
    print()
    print(f'  Winner: {"TabPFN" if t["P50"] < x["P50"] else "XGBoost"} → BiLSTM')
    print('=' * 60)


if __name__ == '__main__':
    main()
