"""TabPFN vs XGBoost: All 5 Resource Dimensions, 5-Fold CV."""
import os, sys, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, load_dist_cache, parse_plan
from tabpfn import TabPFNRegressor
from sklearn.ensemble import GradientBoostingRegressor

ROOT = '/home/anqian/Desktop/my_lab/workloads'
SEED = 42
MAX_LAT = 200.0
N_FOLDS = 5

DIMS = {
    'Latency':  lambda lab: math.log(1 + max(lab['latency_s'], 0.001)),
    'Memory':   lambda lab: math.log(1 + max(lab['memory_bytes'], 1)),
    'DiskIO':   lambda lab: math.log(1 + max(lab['disk_bytes'], 1)),
    'Network':  lambda lab: math.log(1 + max(lab['network_bytes'], 1)),
    'CPU_Res':  lambda lab: math.log(1 + max(lab.get('cpu_resource_s', 0.001), 0.001)),
}

# ---- Load ----
print("=" * 65)
print("TabPFN vs XGBoost: All 5 Resource Dimensions (5-Fold CV)")
print("=" * 65)

ndv = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
dist = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))
cgroup = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))

# Add CPU resource from EXPLAIN ANALYZE
analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
for qid in cgroup:
    af = os.path.join(analyze_dir, f'{qid}.txt')
    if os.path.exists(af):
        with open(af) as f:
            cgroup[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

print("\n[1] Extracting features...")
X_list = []
y_lists = {k: [] for k in DIMS}
n_ok = 0
for qid, lab in cgroup.items():
    if lab.get('latency_s', 999) > MAX_LAT:
        continue
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        continue
    with open(pf) as f:
        plan = f.read()
    g = parse_plan(plan, ndv, dist)
    if g is None or g.x.shape[0] == 0:
        continue
    x = g.x.numpy()
    feats = []
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(x.shape[0])
    feats.append(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0)
    X_list.append(feats)
    for k, fn in DIMS.items():
        y_lists[k].append(fn(lab))
    n_ok += 1

X = np.array(X_list, dtype=np.float64)
for k in DIMS:
    y_lists[k] = np.array(y_lists[k], dtype=np.float64)
print(f"  Final: {n_ok} queries, {X.shape[1]} features")

# ---- 5-Fold CV ----
np.random.seed(SEED)
n = n_ok
idx = np.random.permutation(n)
fs = n // N_FOLDS

# Results: {dim: {model: [fold_qerrors]}}
results = {k: {'XGBoost': [], 'TabPFN': []} for k in DIMS}

for fold in range(N_FOLDS):
    print(f"\n[2] Fold {fold + 1}/{N_FOLDS}")
    te_s = fold * fs
    te_e = (fold + 1) * fs if fold < N_FOLDS - 1 else n
    te_idx = idx[te_s:te_e]
    tr_idx = np.setdiff1d(idx, te_idx)

    X_tr, X_te = X[tr_idx], X[te_idx]
    Xm = X_tr.mean(axis=0)
    Xs = X_tr.std(axis=0) + 1e-8
    X_tr_std = (X_tr - Xm) / Xs
    X_te_std = (X_te - Xm) / Xs

    for dim in DIMS:
        y_all = y_lists[dim]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        ym = y_tr.mean()
        ys = max(y_tr.std(), 1e-8)

        # -- XGBoost --
        xgb = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED + fold, subsample=0.8)
        xgb.fit(X_tr_std, (y_tr - ym) / ys)
        xgb_pz = xgb.predict(X_te_std)
        xgb_p = np.maximum(np.exp(xgb_pz * ys + ym) - 1, 0.01)
        t_raw = np.maximum(np.exp(y_te) - 1, 0.01)
        xgb_qe = np.sort(np.maximum(xgb_p / t_raw, t_raw / xgb_p))
        results[dim]['XGBoost'].append(xgb_qe)

        # -- TabPFN --
        tpf = TabPFNRegressor(
            n_estimators=8, random_state=SEED + fold,
            fit_mode='low_memory', device='cuda')
        tpf.fit(X_tr.astype(np.float32), ((y_tr - ym) / ys).astype(np.float32))
        tpf_pz = tpf.predict(X_te.astype(np.float32)).astype(np.float64)
        tpf_p = np.maximum(np.exp(tpf_pz * ys + ym) - 1, 0.01)
        tpf_qe = np.sort(np.maximum(tpf_p / t_raw, t_raw / tpf_p))
        results[dim]['TabPFN'].append(tpf_qe)

    # Per-fold summary
    print(f"  {'Dim':<10} {'XGB P50':>8} {'TPF P50':>8} {'Delta':>8}")
    for dim in DIMS:
        x50 = results[dim]['XGBoost'][-1][len(results[dim]['XGBoost'][-1]) // 2]
        t50 = results[dim]['TabPFN'][-1][len(results[dim]['TabPFN'][-1]) // 2]
        d = (t50 - x50) / x50 * 100
        print(f"  {dim:<10} {x50:.2f}x   {t50:.2f}x   {d:+.1f}%")

# ---- Final Summary ----
print("\n" + "=" * 65)
print("FINAL RESULTS (5-Fold CV Avg)")
print("=" * 65)

print(f"\n  {'Dim':<10} {'Model':<12} {'P50':>7} {'P90':>7} {'P95':>7} {'Win':>8}")
print("  " + "-" * 52)
for dim in DIMS:
    for model in ['XGBoost', 'TabPFN']:
        folds = results[dim][model]
        p50s = [f[len(f)//2] for f in folds]
        p90s = [f[int(len(f)*0.9)] for f in folds]
        p95s = [f[int(len(f)*0.95)] for f in folds]
        winner = ''
        if model == 'TabPFN':
            x50 = np.mean([results[dim]['XGBoost'][i][len(results[dim]['XGBoost'][i])//2]
                            for i in range(N_FOLDS)])
            t50 = np.mean(p50s)
            winner = 'TPF' if t50 < x50 else 'XGB'
        print(f"  {dim:<10} {model:<12} {np.mean(p50s):.2f}x {np.mean(p90s):.2f}x "
              f"{np.mean(p95s):.2f}x {winner:>8}")

# Overall winner count
xgb_wins = 0; tpf_wins = 0
for dim in DIMS:
    x50 = np.mean([results[dim]['XGBoost'][i][len(results[dim]['XGBoost'][i])//2]
                    for i in range(N_FOLDS)])
    t50 = np.mean([results[dim]['TabPFN'][i][len(results[dim]['TabPFN'][i])//2]
                    for i in range(N_FOLDS)])
    if t50 < x50: tpf_wins += 1
    else: xgb_wins += 1

print(f"\n  Score: TabPFN {tpf_wins} - {xgb_wins} XGBoost")
print("=" * 65)
