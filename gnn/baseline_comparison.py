"""
Baseline comparison: GNN vs XGBoost vs MLP vs ICONQ+MLP for resource prediction.
All use same cgroup labels, same data split, same 5-dim targets.

Usage:
    cd /home/anqian/Desktop/my_lab/workloads
    source /home/anqian/code/python/workloads/venv/bin/activate
    python gnn/baseline_comparison.py --seed 42
"""

import os, sys, re, math, json, argparse, numpy as np, torch, torch.nn as nn
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import PlanGNN
from train_cgroup import load_dataset, normalize_labels, load_cgroup_labels
from train_ndv import load_ndv_cache, load_dist_cache

ROOT = '/home/anqian/Desktop/my_lab/workloads'

# ─── ICONQ-style flat feature extraction ───
OP_TYPES = ['TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
            'IndexLookUp', 'IndexReader', 'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
            'HashAgg', 'StreamAgg', 'Sort', 'TopN', 'Window',
            'ExchangeSender', 'ExchangeReceiver', 'Projection', 'Selection']
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer', 'nation', 'region']


def extract_iconq_features(qid):
    """47-dim ICONQ flat features from EXPLAIN plan."""
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf): return None
    with open(pf) as f: plan = f.read()
    oc = {o: 0 for o in OP_TYPES}; oe = {o: 0.0 for o in OP_TYPES}; tc = {t: 0.0 for t in TABLES}
    for line in plan.split('\n'):
        if '\t' not in line or line.startswith('--'): continue
        parts = line.lstrip(' │├└─').split('\t')
        if len(parts) < 5: continue
        op = re.sub(r'^[│├└─\s]+', '', parts[0].strip())
        op = re.sub(r'\(Build\)|\(Probe\)', '', op).strip()
        op = re.sub(r'_\d+$', '', op)
        try: est = float(parts[1].strip())
        except: est = 1.0
        if op in oc: oc[op] += 1; oe[op] += est
        oi = parts[4].strip() if len(parts) > 4 else ''
        for t in TABLES:
            if t in oi.lower(): tc[t] = max(tc[t], est)
    feat = []
    for o in OP_TYPES: feat.append(float(oc[o])); feat.append(math.log(1 + oe[o]))
    for t in TABLES: feat.append(math.log(1 + tc[t]))
    return feat  # 19*2 + 8 = 46 dim (no predicted latency for fair comparison)


def extract_flat_features(graph):
    """Extract flat statistical features from PyG graph for XGBoost."""
    x = graph.x.numpy()  # [n_nodes, node_dim]
    n_nodes = x.shape[0]
    feats = []
    # Per-column: mean, max, sum, std
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(n_nodes)
    feats.append(graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0)  # n_edges
    return np.array(feats, dtype=np.float32)


def extract_pooled_features(graph):
    """Pool node features: concat(mean, max, sum) for MLP baseline."""
    x = graph.x.numpy()
    return np.concatenate([x.mean(axis=0), x.max(axis=0), x.sum(axis=0)]).astype(np.float32)


# ─── Simple MLP ───
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, n_targets=5, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_targets))

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, stats, label_keys,
              hidden=128, epochs=300, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleMLP(X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    best_val, best_state = float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_tr)
        loss = nn.functional.huber_loss(pred, y_tr)
        loss.backward(); opt.step(); scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va)
                val_loss = nn.functional.huber_loss(val_pred, y_va).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    X_te = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()

    return evaluate_predictions(preds, y_test, stats, label_keys)


def train_xgboost(X_train, y_train, X_test, y_test, stats, label_keys, seed=42):
    from sklearn.ensemble import GradientBoostingRegressor
    results = {}
    for i, k in enumerate(label_keys):
        model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           random_state=seed, subsample=0.8)
        model.fit(X_train, y_train[:, i])
        preds_i = model.predict(X_test)
        # Denormalize
        raw_name = {'mem': 'memory_bytes', 'net': 'network_bytes', 'disk': 'disk_bytes',
                     'lat': 'latency_s', 'cpures': 'cpu_resource_s'}[k]
        p_raw = np.exp(preds_i * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
        t_raw = np.exp(y_test[:, i] * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
        p_raw = np.maximum(p_raw, 0.01); t_raw = np.maximum(t_raw, 0.01)
        qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
        n = len(qe)
        r2 = 1 - np.sum((np.log(1+t_raw) - np.log(1+p_raw))**2) / max(np.sum((np.log(1+t_raw) - np.mean(np.log(1+t_raw)))**2), 1e-8)
        results[k] = {'P50': qe[n//2], 'P90': qe[int(n*0.9)], 'P95': qe[int(n*0.95)], 'R2': r2}
    return results


def evaluate_predictions(preds, y_test, stats, label_keys):
    results = {}
    raw_map = {'mem': 'memory_bytes', 'net': 'network_bytes', 'disk': 'disk_bytes',
               'lat': 'latency_s', 'cpures': 'cpu_resource_s'}
    for i, k in enumerate(label_keys):
        raw_name = raw_map[k]
        p_raw = np.exp(preds[:, i] * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
        t_raw = np.exp(y_test[:, i] * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
        p_raw = np.maximum(p_raw, 0.01); t_raw = np.maximum(t_raw, 0.01)
        qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
        n = len(qe)
        r2 = 1 - np.sum((np.log(1+t_raw) - np.log(1+p_raw))**2) / max(np.sum((np.log(1+t_raw) - np.mean(np.log(1+t_raw)))**2), 1e-8)
        results[k] = {'P50': qe[n//2], 'P90': qe[int(n*0.9)], 'P95': qe[int(n*0.95)], 'R2': r2}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    plan_dir = os.path.join(ROOT, 'explain_plans')
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
    dist_cache = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))

    # Load cgroup labels using the same function as train_cgroup.py
    cgroup_dir = os.path.join(ROOT, 'cgroup_resources')
    cgroup_labels = load_cgroup_labels(cgroup_dir)
    print(f'Cgroup labels: {len(cgroup_labels)}')

    # Load GNN graphs + labels
    graphs, labels, meta = load_dataset(plan_dir, ndv_cache, dist_cache, cgroup_labels)
    print(f'Matched: {len(graphs)} queries')

    # Normalize labels
    norm_labels, stats = normalize_labels(labels)
    label_keys = ['mem', 'net', 'disk', 'lat', 'cpures']
    key_map = {'memory_bytes': 'mem', 'network_bytes': 'net', 'disk_bytes': 'disk',
               'latency_s': 'lat', 'cpu_resource_s': 'cpures'}

    # Same split as train_cgroup.py
    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.7)]
    val_idx = indices[int(n * 0.7):int(n * 0.85)]
    test_idx = indices[int(n * 0.85):]
    print(f'Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    # Build label arrays (norm_labels uses raw keys)
    raw_keys = ['memory_bytes', 'network_bytes', 'disk_bytes', 'latency_s', 'cpu_resource_s']
    y_all = np.array([[nl[rk] for rk in raw_keys] for nl in norm_labels], dtype=np.float32)
    y_train = y_all[train_idx]; y_val = y_all[val_idx]; y_test = y_all[test_idx]

    # ═══ 1. XGBoost on flat features ═══
    print('\n' + '='*70)
    print('1. XGBoost on flat statistical features')
    X_flat = np.array([extract_flat_features(graphs[i]) for i in range(n)])
    # Normalize
    Xm = X_flat[train_idx].mean(axis=0); Xs = X_flat[train_idx].std(axis=0) + 1e-8
    X_flat_n = (X_flat - Xm) / Xs
    xgb_results = train_xgboost(X_flat_n[train_idx], y_train, X_flat_n[test_idx], y_test,
                                 stats, label_keys, seed=args.seed)
    print(f'  Flat features dim: {X_flat.shape[1]}')

    # ═══ 2. MLP on pooled node features ═══
    print('\n' + '='*70)
    print('2. MLP on pooled node features (mean+max+sum)')
    X_pool = np.array([extract_pooled_features(graphs[i]) for i in range(n)])
    Xm2 = X_pool[train_idx].mean(axis=0); Xs2 = X_pool[train_idx].std(axis=0) + 1e-8
    X_pool_n = (X_pool - Xm2) / Xs2
    mlp_pool_results = train_mlp(X_pool_n[train_idx], y_train, X_pool_n[val_idx], y_val,
                                  X_pool_n[test_idx], y_test, stats, label_keys,
                                  hidden=128, epochs=args.epochs, seed=args.seed)
    print(f'  Pooled features dim: {X_pool.shape[1]}')

    # ═══ 3. ICONQ features + MLP ═══
    print('\n' + '='*70)
    print('3. ICONQ flat features + MLP')
    iconq_feats = []
    valid_mask = []
    for i in range(n):
        feat = extract_iconq_features(meta[i])
        if feat is not None:
            iconq_feats.append(feat)
            valid_mask.append(True)
        else:
            iconq_feats.append([0.0] * 46)
            valid_mask.append(False)
    X_iconq = np.array(iconq_feats, dtype=np.float32)
    Xm3 = X_iconq[train_idx].mean(axis=0); Xs3 = X_iconq[train_idx].std(axis=0) + 1e-8
    X_iconq_n = (X_iconq - Xm3) / Xs3
    iconq_results = train_mlp(X_iconq_n[train_idx], y_train, X_iconq_n[val_idx], y_val,
                               X_iconq_n[test_idx], y_test, stats, label_keys,
                               hidden=128, epochs=args.epochs, seed=args.seed)
    print(f'  ICONQ features dim: {X_iconq.shape[1]}')

    # ═══ 4. GNN (from checkpoint) ═══
    print('\n' + '='*70)
    print('4. GNN (GATv2Conv) — loaded from checkpoint')
    from torch_geometric.loader import DataLoader
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in key_map.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(ROOT, 'checkpoints', 'best_cgroup.pt')
    if os.path.exists(ckpt_path):
        model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64)
        all_p = {k: [] for k in label_keys}
        all_t = {k: [] for k in label_keys}
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                preds = model(data)
                for k in label_keys:
                    all_p[k].append(preds[k].squeeze(-1).cpu().numpy())
                    all_t[k].append(getattr(data, f'y_{k}').squeeze(-1).cpu().numpy())

        gnn_results = {}
        raw_map = {'mem': 'memory_bytes', 'net': 'network_bytes', 'disk': 'disk_bytes',
                   'lat': 'latency_s', 'cpures': 'cpu_resource_s'}
        for k in label_keys:
            p = np.concatenate(all_p[k])
            t = np.concatenate(all_t[k])
            raw_name = raw_map[k]
            p_raw = np.exp(p * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
            t_raw = np.exp(t * stats[raw_name]['std'] + stats[raw_name]['mean']) - 1
            p_raw = np.maximum(p_raw, 0.01); t_raw = np.maximum(t_raw, 0.01)
            qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
            n_q = len(qe)
            r2 = 1 - np.sum((np.log(1+t_raw) - np.log(1+p_raw))**2) / max(np.sum((np.log(1+t_raw) - np.mean(np.log(1+t_raw)))**2), 1e-8)
            gnn_results[k] = {'P50': qe[n_q//2], 'P90': qe[int(n_q*0.9)], 'P95': qe[int(n_q*0.95)], 'R2': r2}
    else:
        print(f'  WARNING: checkpoint not found at {ckpt_path}')
        print(f'  Run: python gnn/train_cgroup.py --epochs 400 --seed 42')
        gnn_results = {k: {'P50': 0, 'P90': 0, 'P95': 0, 'R2': 0} for k in label_keys}

    # ═══ Summary ═══
    print('\n' + '='*70)
    print('RESOURCE PREDICTION: MODEL COMPARISON')
    print(f'Seed={args.seed}, {len(test_idx)} test queries, 5-dim cgroup labels')
    print('='*70)

    dim_labels = {'mem': 'Memory', 'disk': 'DiskIO', 'net': 'Network', 'lat': 'Latency', 'cpures': 'CPU_Res'}
    all_methods = [
        ('XGBoost (flat)', xgb_results),
        ('MLP (pooled)', mlp_pool_results),
        ('ICONQ+MLP', iconq_results),
        ('GNN (ours)', gnn_results),
    ]

    for dim_key, dim_name in dim_labels.items():
        print(f'\n  {dim_name}:')
        print(f'    {"Method":<20} {"P50":>6} {"P90":>6} {"P95":>6} {"R²":>6}')
        print('    ' + '-' * 48)
        for method_name, method_results in all_methods:
            r = method_results[dim_key]
            print(f'    {method_name:<20} {r["P50"]:.2f}x {r["P90"]:.2f}x {r["P95"]:.2f}x {r["R2"]:.3f}')

    # Average across dimensions
    print(f'\n  Average across 5 dimensions:')
    print(f'    {"Method":<20} {"Avg P50":>8} {"Avg P90":>8} {"Avg R²":>7}')
    print('    ' + '-' * 45)
    for method_name, method_results in all_methods:
        avg_p50 = np.mean([method_results[k]['P50'] for k in label_keys])
        avg_p90 = np.mean([method_results[k]['P90'] for k in label_keys])
        avg_r2 = np.mean([method_results[k]['R2'] for k in label_keys])
        print(f'    {method_name:<20} {avg_p50:.2f}x {avg_p90:.2f}x {avg_r2:.3f}')


if __name__ == '__main__':
    main()
