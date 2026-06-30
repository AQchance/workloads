"""
Train TabPFN on ALL cgroup data for 5 resource dimensions, save models + predictions.

Usage:
    source ~/.zshrc && source /home/anqian/code/python/workloads/venv/bin/activate
    cd /home/anqian/Desktop/my_lab/workloads
    python gnn/train_tabpfn_full.py --seed 42

Outputs:
    checkpoints/tabpfn_mem.joblib     checkpoints/tabpfn_disk.joblib
    checkpoints/tabpfn_net.joblib     checkpoints/tabpfn_lat.joblib
    checkpoints/tabpfn_cpures.joblib
    checkpoints/tabpfn_resource_cache.json   (predictions for all queries)
    checkpoints/tabpfn_feature_stats.npz     (feature mean/std for normalization)
"""
import os, sys, math, json, time, joblib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, load_dist_cache, parse_plan
from tabpfn import TabPFNRegressor

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

DIM_NAMES = ['mem', 'disk', 'net', 'lat', 'cpures']
DIM_LABEL_KEYS = {
    'mem': 'memory_bytes', 'disk': 'disk_bytes',
    'net': 'network_bytes', 'lat': 'latency_s', 'cpures': 'cpu_resource_s'
}


def extract_flat_features_from_qid(qid, ndv_cache):
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
    feats.append(x.shape[0])
    feats.append(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0)
    return np.array(feats, dtype=np.float64)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=8)
    parser.add_argument('--fit-mode', default='fit_preprocessors',
                        choices=['low_memory', 'fit_preprocessors', 'fit_with_cache'])
    args = parser.parse_args()

    print('=' * 60)
    print('Training TabPFN on Full Cgroup Data (5 resource dimensions)')
    print(f'  n_estimators={args.n_estimators}, fit_mode={args.fit_mode}, seed={args.seed}')
    print('=' * 60)

    # ---- Load Data ----
    print('\n[1/4] Loading data...')
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
    dist_cache = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))
    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))

    # Add CPU resource from EXPLAIN ANALYZE
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

    # Extract features + labels
    X_list, y_lists, qids = [], {k: [] for k in DIM_NAMES}, []
    for qid, lab in cgroup_labels.items():
        feat = extract_flat_features_from_qid(qid, ndv_cache)
        if feat is None:
            continue
        X_list.append(feat)
        for dim in DIM_NAMES:
            raw_key = DIM_LABEL_KEYS[dim]
            raw_val = max(lab.get(raw_key, 1), 1)
            y_lists[dim].append(math.log(1 + raw_val))
        qids.append(qid)

    X = np.array(X_list, dtype=np.float64)
    for dim in DIM_NAMES:
        y_lists[dim] = np.array(y_lists[dim], dtype=np.float64)

    print(f'  Training data: {len(X)} queries, {X.shape[1]} features')
    for dim in DIM_NAMES:
        ys = y_lists[dim]
        print(f'    {dim}: y [{ys.min():.2f}, {ys.mean():.2f}, {ys.max():.2f}]')

    # ---- Train TabPFN ----
    print(f'\n[2/4] Training TabPFN (n_estimators={args.n_estimators})...')

    models = {}
    for dim in DIM_NAMES:
        print(f'  Training {dim}...')
        t0 = time.time()
        model = TabPFNRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            fit_mode=args.fit_mode,
            device='cuda',
        )
        model.fit(X.astype(np.float32), y_lists[dim].astype(np.float32))
        elapsed = time.time() - t0
        print(f'    Done in {elapsed:.1f}s')

        # Save model
        path = os.path.join(CKPT_DIR, f'tabpfn_{dim}.joblib')
        joblib.dump(model, path)
        print(f'    Saved to {path}')
        models[dim] = model

    # ---- Cache Predictions ----
    print('\n[3/4] Caching predictions for all queries...')

    resource_cache = {}
    for i, qid in enumerate(qids):
        feat = X[i:i+1].astype(np.float32)
        preds = {}
        for dim in DIM_NAMES:
            preds[dim] = float(models[dim].predict(feat)[0])
        resource_cache[qid] = preds

    cache_path = os.path.join(CKPT_DIR, 'tabpfn_resource_cache.json')
    with open(cache_path, 'w') as f:
        json.dump(resource_cache, f)
    print(f'  Saved: {cache_path} ({len(resource_cache)} queries)')

    # Save feature stats (for potential standardization needs)
    Xm = X.mean(axis=0)
    Xs = X.std(axis=0) + 1e-8
    stats_path = os.path.join(CKPT_DIR, 'tabpfn_feature_stats.npz')
    np.savez(stats_path, mean=Xm, std=Xs, qids=np.array(qids))
    print(f'  Saved: {stats_path}')

    # ---- Quick Validation ----
    print('\n[4/4] Quick validation (5 random queries)...')
    np.random.seed(args.seed)
    val_idx = np.random.choice(len(X), min(5, len(X)), replace=False)
    for i in val_idx:
        qid = qids[i]
        feat = X[i:i+1].astype(np.float32)
        print(f'  {qid}:', end='')
        for dim in DIM_NAMES:
            pred_log = float(models[dim].predict(feat)[0])
            true_log = y_lists[dim][i]
            pred_raw = max(math.exp(pred_log) - 1, 0.01)
            true_raw = max(math.exp(true_log) - 1, 0.01)
            qe = max(pred_raw / true_raw, true_raw / pred_raw)
            print(f'  {dim}={qe:.2f}x', end='')
        print()

    print('\n' + '=' * 60)
    print('DONE. Models saved, ready for Bi-LSTM pipeline.')
    print('=' * 60)


if __name__ == '__main__':
    main()
