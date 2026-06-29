"""TabPFN vs XGBoost: Latency-only 5-Fold CV (fast focused comparison)."""
import os, sys, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cgroup import load_cgroup_labels
from train_ndv import load_ndv_cache, load_dist_cache, parse_plan
from tabpfn import TabPFNRegressor
from sklearn.ensemble import GradientBoostingRegressor

ROOT = '/home/anqian/Desktop/my_lab/workloads'
SEED = 42
MAX_LAT = 200.0
N_FOLDS = 5

# ---- Load ----
print("=" * 60)
print("TabPFN vs XGBoost: Latency Prediction (5-Fold CV)")
print("=" * 60)

ndv = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
dist = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))
cgroup = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))

print("\n[1] Extracting features...")
X_list, y_list = [], []
n_total, n_miss, n_filtered = 0, 0, 0
for qid, lab in cgroup.items():
    n_total += 1
    if lab.get('latency_s', 999) > MAX_LAT:
        n_filtered += 1
        continue
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        n_miss += 1
        continue
    with open(pf) as f:
        plan = f.read()
    g = parse_plan(plan, ndv, dist)
    if g is None or g.x.shape[0] == 0:
        n_miss += 1
        continue
    x = g.x.numpy()
    feats = []
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(x.shape[0])
    feats.append(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0)
    X_list.append(feats)
    y_list.append(math.log(1 + max(lab['latency_s'], 0.001)))

X = np.array(X_list, dtype=np.float64)
y = np.array(y_list, dtype=np.float64)
print(f"  Total: {n_total}, filtered: {n_filtered}, miss: {n_miss}")
print(f"  Final: {len(X)} queries, {X.shape[1]} features")
print(f"  y: [{y.min():.2f}, {y.mean():.2f}, {y.max():.2f}]")

# ---- 5-Fold CV ----
print(f"\n[2] {N_FOLDS}-Fold CV...")
np.random.seed(SEED)
n = len(X)
idx = np.random.permutation(n)
fs = n // N_FOLDS

fold_results_xgb = []
fold_results_tabpfn = []

for fold in range(N_FOLDS):
    print(f"\n  --- Fold {fold + 1}/{N_FOLDS} ---")
    te_start = fold * fs
    te_end = (fold + 1) * fs if fold < N_FOLDS - 1 else n
    te_idx = idx[te_start:te_end]
    tr_idx = np.setdiff1d(idx, te_idx)
    print(f"    train={len(tr_idx)}, test={len(te_idx)}")

    X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Standardize features (for XGBoost)
    Xm = X_tr_raw.mean(axis=0)
    Xs = X_tr_raw.std(axis=0) + 1e-8
    X_tr_std = (X_tr_raw - Xm) / Xs
    X_te_std = (X_te_raw - Xm) / Xs

    # Z-score target
    ym = y_tr.mean()
    ys = max(y_tr.std(), 1e-8)
    y_tr_z = (y_tr - ym) / ys

    # -- XGBoost --
    t0 = time.time()
    xgb = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED + fold, subsample=0.8)
    xgb.fit(X_tr_std, y_tr_z)
    xgb_pred = xgb.predict(X_te_std)
    xgb_t = time.time() - t0

    # Denorm XGBoost
    y_te_norm = (y_te - ym) / ys
    xgb_p = np.exp(xgb_pred * ys + ym) - 1
    t_raw = np.exp(y_te_norm * ys + ym) - 1
    xgb_p = np.maximum(xgb_p, 0.01)
    t_raw = np.maximum(t_raw, 0.01)
    xgb_qe = np.sort(np.maximum(xgb_p / t_raw, t_raw / xgb_p))
    xgb_r2 = 1 - np.sum((y_te_norm - xgb_pred) ** 2) / max(
        np.sum((y_te_norm - np.mean(y_te_norm)) ** 2), 1e-8)
    nq = len(xgb_qe)
    print(f"    XGBoost: P50={xgb_qe[nq//2]:.2f}x "
          f"P90={xgb_qe[int(nq*0.9)]:.2f}x "
          f"R2={xgb_r2:.3f} ({xgb_t:.1f}s)")
    fold_results_xgb.append((xgb_qe, xgb_r2))

    # -- TabPFN --
    t0 = time.time()
    tpf = TabPFNRegressor(
        n_estimators=8, random_state=SEED + fold,
        fit_mode='low_memory', device='cuda')
    tpf.fit(X_tr_raw.astype(np.float32), y_tr_z.astype(np.float32))
    tpf_pred = tpf.predict(X_te_raw.astype(np.float32)).astype(np.float64)
    tpf_t = time.time() - t0

    tpf_p = np.exp(tpf_pred * ys + ym) - 1
    tpf_p = np.maximum(tpf_p, 0.01)
    tpf_qe = np.sort(np.maximum(tpf_p / t_raw, t_raw / tpf_p))
    tpf_r2 = 1 - np.sum((y_te_norm - tpf_pred) ** 2) / max(
        np.sum((y_te_norm - np.mean(y_te_norm)) ** 2), 1e-8)
    print(f"    TabPFN:  P50={tpf_qe[nq//2]:.2f}x "
          f"P90={tpf_qe[int(nq*0.9)]:.2f}x "
          f"R2={tpf_r2:.3f} ({tpf_t:.1f}s)")
    fold_results_tabpfn.append((tpf_qe, tpf_r2))

# ---- Summary ----
print("\n" + "=" * 60)
print("FINAL RESULTS (Latency, 5-Fold CV)")
print("=" * 60)

header = f"  {'Fold':<8} {'XGB P50':>8} {'TPF P50':>8} {'XGB R2':>8} {'TPF R2':>8} {'Delta':>8}"
print(header)
print("  " + "-" * 48)
for i in range(N_FOLDS):
    xgb_qe, xgb_r2 = fold_results_xgb[i]
    tpf_qe, tpf_r2 = fold_results_tabpfn[i]
    nx = len(xgb_qe)
    delta = (tpf_qe[nx//2] - xgb_qe[nx//2]) / xgb_qe[nx//2] * 100
    print(f"  Fold {i+1}:  {xgb_qe[nx//2]:.2f}x   {tpf_qe[nx//2]:.2f}x   "
          f"{xgb_r2:.3f}   {tpf_r2:.3f}   {delta:+.1f}%")

all_xgb_p50 = [r[0][len(r[0])//2] for r in fold_results_xgb]
all_tpf_p50 = [r[0][len(r[0])//2] for r in fold_results_tabpfn]
all_xgb_p90 = [r[0][int(len(r[0])*0.9)] for r in fold_results_xgb]
all_tpf_p90 = [r[0][int(len(r[0])*0.9)] for r in fold_results_tabpfn]
all_xgb_r2 = [r[1] for r in fold_results_xgb]
all_tpf_r2 = [r[1] for r in fold_results_tabpfn]

print(f"\n  {'Model':<15} {'P50':>8} {'P90':>8} {'R2':>8}")
print("  " + "-" * 35)
print(f"  {'XGBoost':<15} {np.mean(all_xgb_p50):.2f}x "
      f"{np.mean(all_xgb_p90):.2f}x {np.mean(all_xgb_r2):.3f}")
print(f"  {'TabPFN':<15} {np.mean(all_tpf_p50):.2f}x "
      f"{np.mean(all_tpf_p90):.2f}x {np.mean(all_tpf_r2):.3f}")

delta_p50 = ((np.mean(all_tpf_p50) - np.mean(all_xgb_p50))
             / np.mean(all_xgb_p50) * 100)
winner = "TabPFN WINS!" if delta_p50 < 0 else "XGBoost wins"
print(f"\n  TabPFN vs XGBoost: DP50 = {delta_p50:+.1f}%  <- {winner}")
print("=" * 60)
