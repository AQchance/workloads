"""
Final Pipeline: 5-Fold CV XGBoost → BiLSTM + Resource Noise Augmentation.
No data leakage: XGBoost predictions are out-of-fold.
80/10/10 query-level split for BiLSTM.

Usage:
    cd /home/anqian/Desktop/my_lab/workloads
    source /home/anqian/code/python/workloads/venv/bin/activate
    python lstm/train_final_pipeline.py --seed 42
    python lstm/train_final_pipeline.py --seed 42 123 456
"""

import os, sys, json, csv, math, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = '/home/anqian/Desktop/my_lab/workloads'
OUT_DIR = os.path.join(ROOT, 'lstm', 'final_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, 'gnn'))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, parse_plan


# ═══════════════════════ Feature Extraction ═══════════════════════

def extract_flat_features(qid, ndv_cache):
    """90-dim flat features from EXPLAIN plan via PyG graph node statistics."""
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        return None
    with open(pf) as f:
        plan_text = f.read()
    g = parse_plan(plan_text, ndv_cache)
    if g is None or g.x.shape[0] == 0:
        return None
    x = g.x.numpy()
    feats = []
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(x.shape[0])  # n_nodes
    feats.append(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0)  # n_edges
    return np.array(feats, dtype=np.float32)


# ═══════════════════════ 5-Fold CV XGBoost ═══════════════════════

def train_xgboost_5fold(X_cgroup, y_cgroup, qids_cgroup, target_qids, seed=42):
    """
    Train XGBoost with 5-Fold CV on ALL cgroup data.
    Returns out-of-fold predictions for target_qids (zero data leakage).
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = np.zeros_like(y_cgroup)
    dim_names = ['mem', 'disk', 'net', 'lat', 'cpures']

    print(f'  5-Fold CV XGBoost ({len(X_cgroup)} queries)...')
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cgroup)):
        Xm = X_cgroup[train_idx].mean(axis=0)
        Xs = X_cgroup[train_idx].std(axis=0) + 1e-8
        X_tr_n = (X_cgroup[train_idx] - Xm) / Xs
        X_va_n = (X_cgroup[val_idx] - Xm) / Xs
        for i in range(5):
            m = GradientBoostingRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.1,
                random_state=seed, subsample=0.8)
            m.fit(X_tr_n, y_cgroup[train_idx, i])
            oof_preds[val_idx, i] = m.predict(X_va_n)
        print(f'    Fold {fold + 1}: train={len(train_idx)}, val={len(val_idx)}')

    # Build cache for target queries
    qid_to_idx = {q: i for i, q in enumerate(qids_cgroup)}
    oof_cache = {}
    for qid in target_qids:
        if qid not in qid_to_idx:
            continue
        idx = qid_to_idx[qid]
        oof_cache[qid] = {name: float(oof_preds[idx, j])
                          for j, name in enumerate(dim_names)}

    # Report OOF resource prediction accuracy
    target_mask = np.array([q in target_qids for q in qids_cgroup])
    print(f'  OOF resource prediction ({target_mask.sum()} target queries):')
    for j, name in enumerate(dim_names):
        p = np.exp(oof_preds[target_mask, j])
        t = np.exp(y_cgroup[target_mask, j])
        qe = np.sort(np.maximum(p / np.maximum(t, 0.01), np.maximum(t, 0.01) / np.maximum(p, 0.01)))
        n = len(qe)
        print(f'    {name:<8} P50={qe[n // 2]:.2f}x  P90={qe[int(n * 0.9)]:.2f}x')

    return oof_cache


# ═══════════════════════ Sequence Building ═══════════════════════

def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_sequences(data_list, cache, qid_info):
    """Build 19-dim concurrent sequences from resource predictions."""
    X, y_ratio = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if sti == 'penalty':
            continue
        ri = cache.get(qi)
        if ri is None:
            continue
        serial_lat = max(math.exp(ri['lat']) - 1, 0.5)
        qv = [ri['mem'], ri['disk'], ri['net'], ri['lat'], ri['cpures'],
              math.log(1 + serial_lat)]
        tr_ = [ri['mem'], ri['disk'], ri['net'], ri['lat'], ri['cpures']]
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov
                 if oq in qid_info and oq in cache]
        peers.sort()
        for osv, oq in peers:
            rj = cache[oq]
            ovv = [rj['mem'], rj['disk'], rj['net'], rj['lat'], rj['cpures'],
                   rj['lat']]
            oc = [rj['mem'], rj['disk'], rj['net'], rj['lat'], rj['cpures']]
            c = resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    return X, y_ratio


# ═══════════════════════ Data Normalization ═══════════════════════

class RatioDataset(torch.utils.data.Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.lengths[i], self.y[i]


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


def pad_and_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    """Pad sequences and normalize. Compute stats from data if not provided."""
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
        Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))

    X_norm = (Xa - Xm) / Xs
    y_norm = (yl - ym) / ys
    return X_norm, lens, y_norm, Xm, Xs, ym, ys


# ═══════════════════════ Model ═══════════════════════

class BiLSTM_RNA(nn.Module):
    """BiLSTM with Resource Noise Augmentation.

    During training, randomly perturbs resource features to improve
    robustness against XGBoost prediction errors.

    Input layout (19-dim per timestep):
      [0:5]   target resources (mem, disk, net, lat, cpures)
      [5]     target serial_lat_log
      [6:11]  peer resources
      [11]    peer serial_lat_log
      [12]    time_delta
      [13]    is_before
      [14:19] resource_conflict (5-dim)
    """
    def __init__(self, input_dim=19, hidden_dim=256, num_layers=3,
                 dropout=0.15, noise_level=0.05):
        super().__init__()
        self.noise_level = noise_level
        self.emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        if self.training and self.noise_level > 0:
            X = X.clone()
            X[:, :, 0:5] += torch.randn_like(X[:, :, 0:5]) * self.noise_level
            X[:, :, 6:11] += torch.randn_like(X[:, :, 6:11]) * self.noise_level
            X[:, :, 14:19] += torch.randn_like(X[:, :, 14:19]) * self.noise_level * 0.5
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.pred(final).squeeze(-1)


# ═══════════════════════ Training ═══════════════════════

def train_bilstm(train_ds, val_ds, test_ds, ym, ys, input_dim,
                 seed=42, epochs=400, lr=5e-4, noise_level=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiLSTM_RNA(input_dim=input_dim, noise_level=noise_level).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    best_val, best_state = float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sch.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            ap, at = [], []
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
                print(f'    E{epoch:3d} val={med:.3f}x best={best_val:.3f}x')

    model.load_state_dict(best_state)
    model.eval()
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
    return model, best_state, qe, n_params


# ═══════════════════════ Main ═══════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='+', default=[42],
                        help='One or more seeds (e.g., --seed 42 123 456)')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Seeds: {args.seed}, Epochs: {args.epochs}, Noise: {args.noise}, Folds: {args.n_folds}')

    # ─── Load data ───
    print(f'\n{"=" * 60}\nLoading data...')
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))

    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())
    print(f'  Cgroup labels: {len(cgroup_labels)}')

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
                timeline.append((float(row['start']), float(row['start']) + actual,
                                 row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    gnf = {}
    for fn in ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
               'lstm/gnn_features_k4_fixed.json']:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))
    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))
    print(f'  Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # ─── Prepare XGBoost features ───
    label_keys = ['memory_bytes', 'disk_bytes', 'network_bytes', 'latency_s', 'cpu_resource_s']
    X_cgroup, y_cgroup, qids_cgroup = [], [], []
    for qid, lab in cgroup_labels.items():
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None:
            continue
        X_cgroup.append(feat)
        y_cgroup.append([math.log(1 + max(lab[k], 0.001)) for k in label_keys])
        qids_cgroup.append(qid)
    X_cgroup = np.array(X_cgroup)
    y_cgroup = np.array(y_cgroup)
    qids_cgroup = np.array(qids_cgroup)

    # ─── 5-Fold CV XGBoost (no data leakage) ───
    print(f'\n{"=" * 60}\nXGBoost 5-Fold Cross-Validation...')
    oof_cache = train_xgboost_5fold(
        X_cgroup, y_cgroup, qids_cgroup, set(unique_qids), seed=42)
    print(f'  OOF cache: {len(oof_cache)} queries')

    # ─── Build concurrent sets ───
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))

    # ─── Run for each seed ───
    all_results = []
    pcts = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]

    for seed in args.seed:
        print(f'\n{"=" * 60}')
        print(f'Seed {seed}')
        print(f'{"=" * 60}')

        # 80/10/10 query-level split
        np.random.seed(seed)
        qids_shuffled = unique_qids.copy()
        np.random.shuffle(qids_shuffled)
        n_tr = int(len(qids_shuffled) * 0.8)
        n_va = int(len(qids_shuffled) * 0.1)
        train_qids = set(qids_shuffled[:n_tr])
        val_qids = set(qids_shuffled[n_tr:n_tr + n_va])
        test_qids = set(qids_shuffled[n_tr + n_va:])

        train_d = [c for c in ci if c[0] in train_qids]
        val_d = [c for c in ci if c[0] in val_qids]
        test_d = [c for c in ci if c[0] in test_qids]

        X_tr, y_tr = build_sequences(train_d, oof_cache, qid_info)
        X_va, y_va = build_sequences(val_d, oof_cache, qid_info)
        X_te, y_te = build_sequences(test_d, oof_cache, qid_info)

        d = len(X_tr[0][0])
        ml = max(max(len(s) for s in X_tr),
                 max(len(s) for s in X_va),
                 max(len(s) for s in X_te))

        print(f'  Data: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test, '
              f'dim={d}, max_len={ml}')

        # Normalize (train stats only)
        Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_and_normalize(X_tr, y_tr, ml)
        Xn_va, l_va, yn_va, _, _, _, _ = pad_and_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
        Xn_te, l_te, yn_te, _, _, _, _ = pad_and_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

        train_ds = RatioDataset(Xn_tr, l_tr, yn_tr)
        val_ds = RatioDataset(Xn_va, l_va, yn_va)
        test_ds = RatioDataset(Xn_te, l_te, yn_te)

        # Train BiLSTM + RNA
        print(f'  Training BiLSTM + RNA (noise={args.noise})...')
        model, state, qe, n_params = train_bilstm(
            train_ds, val_ds, test_ds, ym, ys, d,
            seed=seed, epochs=args.epochs, noise_level=args.noise)

        n = len(qe)
        result = {
            'seed': seed, 'n_params': n_params, 'n_test': n,
            'metrics': {f'P{p}': round(float(qe[min(int(n * p / 100), n - 1)]), 2)
                        for p in pcts}
        }
        all_results.append(result)

        print(f'\n  Seed {seed}: P50={qe[n // 2]:.3f}x  P90={qe[int(n * 0.9)]:.3f}x  '
              f'P95={qe[int(n * 0.95)]:.3f}x  ({n_params:,} params)')

        # Save model
        torch.save(state, os.path.join(OUT_DIR, f'bilstm_rna_s{seed}.pt'))

    # ─── Summary ───
    print(f'\n{"=" * 60}')
    print(f'FINAL RESULTS: 5-Fold CV XGBoost + BiLSTM + RNA(noise={args.noise})')
    print(f'{"=" * 60}')
    print(f'{"Seed":>6} {"P50":>7} {"P90":>7} {"P95":>7}')
    print('-' * 30)
    for r in all_results:
        m = r['metrics']
        print(f'{r["seed"]:>6} {m["P50"]:.2f}x {m["P90"]:.2f}x {m["P95"]:.2f}x')

    if len(all_results) > 1:
        avg_p50 = np.mean([r['metrics']['P50'] for r in all_results])
        avg_p90 = np.mean([r['metrics']['P90'] for r in all_results])
        avg_p95 = np.mean([r['metrics']['P95'] for r in all_results])
        print(f'{"Mean":>6} {avg_p50:.2f}x {avg_p90:.2f}x {avg_p95:.2f}x')
        print(f'\nICONQ ref: P50=~1.48x  P90=~3.05x  P95=~4.08x')
        print(f'Ours vs ICONQ: P50 {(1.48 - avg_p50) / 1.48 * 100:+.1f}%  '
              f'P90 {(3.05 - avg_p90) / 3.05 * 100:+.1f}%  '
              f'P95 {(4.08 - avg_p95) / 4.08 * 100:+.1f}%')

    # Save results JSON
    results_path = os.path.join(OUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'epochs': args.epochs, 'noise_level': args.noise,
                'n_folds': args.n_folds, 'split': '80/10/10',
                'xgboost': 'n=500 depth=4 lr=0.1',
                'bilstm': 'h=256 layers=3 dropout=0.15 lr=5e-4',
            },
            'results': all_results,
        }, f, indent=2)
    print(f'\nResults saved: {results_path}')
    print(f'Models saved: {OUT_DIR}/bilstm_rna_s*.pt')


if __name__ == '__main__':
    main()
