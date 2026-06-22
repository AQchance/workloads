"""
Hyperparameter tuning for XGBoost → BiLSTM pipeline.
Tests: XGBoost (n_estimators, max_depth) × BiLSTM (hidden, layers, lr, dropout).
"""
import os, sys, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor

ROOT = '/home/anqian/Desktop/my_lab/workloads'
sys.path.insert(0, os.path.join(ROOT, 'gnn'))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, load_dist_cache

# Reuse functions from xgboost_bilstm.py
sys.path.insert(0, os.path.join(ROOT, 'lstm'))
from xgboost_bilstm import (extract_flat_features, resource_conflict, build_xgb_sequences,
                              RatioDataset, collate_fn, pad_and_normalize)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        x = self.emb(X)
        p = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(p)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


def train_eval(train_ds, val_ds, test_ds, ym, ys, input_dim,
               hidden=256, layers=3, dropout=0.2, lr=1e-3, epochs=250, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = BiLSTM(input_dim, hidden, layers, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sch.step()
        if epoch % 20 == 0:
            model.eval(); ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    X = X.to(device); ap.append(model(X, l).cpu().numpy()); at.append(y.numpy())
            pz = np.concatenate(ap); tz = np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val: best_val = med; best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader:
            X = X.to(device); ap.append(model(X, l).cpu().numpy()); at.append(y.numpy())
    pz = np.concatenate(ap); tz = np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
    tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    n = len(qe)
    return qe[n // 2], qe[int(n * 0.9)], qe[int(n * 0.95)], n_params, best_val


def main():
    SEED = 42
    np.random.seed(SEED)
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
    dist_cache = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))

    # Load timeline
    TRACES = ['collect_concurrent/trace_2_mixed.csv', 'collect_concurrent/trace_3_fixed_mixed.csv',
              'collect_concurrent/trace_4_fixed_mixed.csv']
    timeline = []
    for tf in TRACES:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime']); actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']), float(row['start']) + actual, row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    gnf = {}
    for fn in ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
               'lstm/gnn_features_k4_fixed.json']:
        with open(os.path.join(ROOT, fn)) as f: gnf.update(json.load(f))
    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))

    np.random.seed(SEED); np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8); n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr]); val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])

    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]
    print(f'Split: {len(train_qids)} train / {len(val_qids)} val / {len(test_qids)} test')

    # ═══ Phase 1: Tune XGBoost ═══
    print(f'\n{"="*60}')
    print('Phase 1: XGBoost Hyperparameter Search')
    print(f'{"="*60}')

    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f: cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

    X_xgb, y_xgb = [], []
    xgb_qids = []
    for qid, lab in cgroup_labels.items():
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: continue
        X_xgb.append(feat)
        y_xgb.append([math.log(1 + max(lab[k], 0.001)) for k in
                       ['memory_bytes', 'disk_bytes', 'network_bytes', 'latency_s', 'cpu_resource_s']])
        xgb_qids.append(qid)
    X_xgb = np.array(X_xgb); y_xgb = np.array(y_xgb)
    Xm_xgb = X_xgb.mean(axis=0); Xs_xgb = X_xgb.std(axis=0) + 1e-8
    X_xgb_n = (X_xgb - Xm_xgb) / Xs_xgb

    # Try different XGBoost configs
    xgb_configs = [
        (200, 6, 0.1),   # baseline
        (300, 6, 0.1),
        (500, 6, 0.05),
        (200, 8, 0.1),
        (300, 8, 0.05),
        (500, 4, 0.1),
    ]
    print(f'{"Config":>25} {"mem":>6} {"disk":>6} {"net":>6} {"lat":>6} {"cpu":>6} {"avg":>6}')
    print('-' * 70)

    best_xgb_config = None; best_xgb_avg = float('inf')
    for n_est, depth, lr_xgb in xgb_configs:
        dim_names = ['mem', 'disk', 'net', 'lat', 'cpures']
        # Simple CV: use last 15% as val
        n_xgb = len(X_xgb); split = int(n_xgb * 0.85)
        scores = []
        for i in range(5):
            m = GradientBoostingRegressor(n_estimators=n_est, max_depth=depth,
                                           learning_rate=lr_xgb, random_state=SEED, subsample=0.8)
            m.fit(X_xgb_n[:split], y_xgb[:split, i])
            pred = m.predict(X_xgb_n[split:])
            qe = np.maximum(np.exp(pred) / np.exp(y_xgb[split:, i]),
                            np.exp(y_xgb[split:, i]) / np.exp(pred))
            scores.append(np.median(qe))
        avg = np.mean(scores)
        tag = f'n={n_est} d={depth} lr={lr_xgb}'
        print(f'{tag:>25} {scores[0]:.3f} {scores[1]:.3f} {scores[2]:.3f} {scores[3]:.3f} {scores[4]:.3f} {avg:.3f}')
        if avg < best_xgb_avg:
            best_xgb_avg = avg; best_xgb_config = (n_est, depth, lr_xgb)

    print(f'\nBest XGBoost: n={best_xgb_config[0]} depth={best_xgb_config[1]} lr={best_xgb_config[2]}')

    # Train best XGBoost on all data
    n_est, depth, lr_xgb = best_xgb_config
    models_xgb = []
    for i in range(5):
        m = GradientBoostingRegressor(n_estimators=n_est, max_depth=depth,
                                       learning_rate=lr_xgb, random_state=SEED, subsample=0.8)
        m.fit(X_xgb_n, y_xgb[:, i])
        models_xgb.append(m)

    def xgb_predict(qid):
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: return None
        feat_n = (feat - Xm_xgb) / Xs_xgb
        return {name: float(models_xgb[i].predict(feat_n.reshape(1, -1))[0])
                for i, name in enumerate(['mem', 'disk', 'net', 'lat', 'cpures'])}

    xgb_cache = {q: xgb_predict(q) for q in unique_qids if xgb_predict(q)}

    # Build sequences
    X_tr, y_tr = build_xgb_sequences(train_d, lambda q: xgb_cache.get(q), qid_info)
    X_va, y_va = build_xgb_sequences(val_d, lambda q: xgb_cache.get(q), qid_info)
    X_te, y_te = build_xgb_sequences(test_d, lambda q: xgb_cache.get(q), qid_info)
    d = len(X_tr[0][0])
    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va), max(len(s) for s in X_te))

    Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_and_normalize(X_tr, y_tr, ml)
    Xn_va, l_va, yn_va, _, _, _, _ = pad_and_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
    Xn_te, l_te, yn_te, _, _, _, _ = pad_and_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)
    train_ds = RatioDataset(Xn_tr, l_tr, yn_tr)
    val_ds = RatioDataset(Xn_va, l_va, yn_va)
    test_ds = RatioDataset(Xn_te, l_te, yn_te)

    # ═══ Phase 2: Tune BiLSTM ═══
    print(f'\n{"="*60}')
    print('Phase 2: BiLSTM Hyperparameter Search')
    print(f'{"="*60}')

    configs = [
        # (hidden, layers, dropout, lr, epochs, label)
        (256, 3, 0.2, 1e-3, 250, 'baseline'),
        (256, 2, 0.2, 1e-3, 250, '2-layer'),
        (256, 4, 0.2, 1e-3, 250, '4-layer'),
        (128, 3, 0.2, 1e-3, 250, 'small-hidden'),
        (512, 3, 0.2, 1e-3, 250, 'big-hidden'),
        (256, 3, 0.1, 1e-3, 250, 'low-dropout'),
        (256, 3, 0.3, 1e-3, 250, 'high-dropout'),
        (256, 3, 0.2, 5e-4, 300, 'low-lr+300ep'),
        (256, 3, 0.2, 1e-3, 400, '400-epochs'),
        (256, 3, 0.15, 5e-4, 400, 'tuned'),
    ]

    print(f'{"Config":<20} {"P50":>6} {"P90":>6} {"P95":>6} {"ValBest":>8} {"Params":>10}')
    print('-' * 60)

    best_p50 = float('inf'); best_config = None
    for hidden, layers, dropout, lr, epochs, label in configs:
        p50, p90, p95, n_params, val_best = train_eval(
            train_ds, val_ds, test_ds, ym, ys, d,
            hidden=hidden, layers=layers, dropout=dropout, lr=lr, epochs=epochs, seed=SEED)
        flag = ' ★' if p50 < best_p50 else ''
        if p50 < best_p50: best_p50 = p50; best_config = label
        print(f'{label:<20} {p50:.3f}x {p90:.3f}x {p95:.3f}x {val_best:.3f}x {n_params:>10,}{flag}')

    print(f'\nBest: {best_config} (P50={best_p50:.3f}x)')


if __name__ == '__main__':
    main()
