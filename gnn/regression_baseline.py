"""Compare GNN vs linear regression for memory prediction."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ndv import load_dataset, load_ndv_cache, load_dist_cache

base = '/home/anqian/Desktop/my_lab/workloads'
ndv = load_ndv_cache(f'{base}/ndv_cache.json')
dist = load_dist_cache(f'{base}/dist_cache.json')
graphs, labels, meta = load_dataset(f'{base}/explain_plans', f'{base}/explain_analyze_results', ndv, dist)

# Clip extreme memory values (CARTESIAN join artifacts)
MAX_MEM = 500e9  # cap at 500GB
y_raw = np.array([min(labels[i]['memory_bytes'], MAX_MEM) for i in range(len(graphs))])
y_raw = np.maximum(y_raw, 1.0)
y = np.log(y_raw)
print(f"y (log memory): [{y.min():.1f}, {y.mean():.1f}, {y.max():.1f}] "
      f"→ exp=[{np.exp(y.min())/1e6:.1f}MB, {np.exp(y.max())/1e9:.1f}GB]")

# Debug: show a few raw memory values
mem_raw = np.array([labels[i]['memory_bytes'] for i in range(len(graphs))])
print(f"raw memory: min={mem_raw.min()/1e6:.1f}MB, median={np.median(mem_raw)/1e9:.2f}GB, "
      f"max={mem_raw.max()/1e9:.1f}GB")

# Check the three NDV memory proxy features
X1 = np.array([graphs[i].x[:, 13].sum().item() for i in range(len(graphs))])
X2 = np.array([graphs[i].x[:, 14].sum().item() for i in range(len(graphs))])
X3 = np.array([graphs[i].x[:, 15].sum().item() for i in range(len(graphs))])
print(f"\nFeatures: join_mem_log [{X1.min():.1f}, {X1.mean():.1f}, {X1.max():.1f}]")
print(f"          agg_mem_log  [{X2.min():.1f}, {X2.mean():.1f}, {X2.max():.1f}]")
print(f"          sort_mem_log [{X3.min():.1f}, {X3.mean():.1f}, {X3.max():.1f}]")

# Correlation with log memory
print(f"\nCorrelation with log(memory):")
print(f"  join_mem_log: r={np.corrcoef(X1, y)[0,1]:.4f}")
print(f"  agg_mem_log:  r={np.corrcoef(X2, y)[0,1]:.4f}")
print(f"  sort_mem_log: r={np.corrcoef(X3, y)[0,1]:.4f}")

# What about sum of all three?
X_sum = X1 + X2 + X3
print(f"  sum_all:      r={np.corrcoef(X_sum, y)[0,1]:.4f}")

# Simple linear regression: log(mem) ~ sum of NDV proxies
np.random.seed(42)
n = len(graphs)
idx = np.random.permutation(n)
tr, te = idx[:int(n*0.7)], idx[int(n*0.85):]
X_tr, X_te = X_sum[tr].reshape(-1,1), X_sum[te].reshape(-1,1)
y_tr, y_te = y[tr], y[te]

# Fit OLS with intercept
X_tr_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
w = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)[0]
bias, w1 = w[0], w[1]
yp = bias + X_te.flatten() * w1

print(f"\nOLS: log(mem) = {bias:.3f} + {w1:.3f} * sum_ndv_proxies")
print(f"Test pred range: [{yp.min():.1f}, {yp.mean():.1f}, {yp.max():.1f}]")
print(f"Test true range: [{y_te.min():.1f}, {y_te.mean():.1f}, {y_te.max():.1f}]")

yp_r = np.maximum(np.exp(yp) - 1, 0)
yt_r = np.exp(y_te) - 1
qe = np.maximum(yp_r / np.maximum(yt_r, 1), np.maximum(yt_r, 1) / np.maximum(yp_r, 1))
qs = np.sort(qe); nq = len(qs)
r2 = 1 - np.sum((y_te-yp)**2) / max(np.sum((y_te-np.mean(y_te))**2), 1e-8)

print(f"\n=== OLS: log(mem) ~ sum(NDV proxies) ===")
print(f"  P50={qs[nq//2]:.2f}x  P90={qs[int(nq*0.9)]:.2f}x  P95={qs[int(nq*0.95)]:.2f}x  R²={r2:.4f}")
print(f"  <2x: {np.mean(qe<=2)*100:.1f}%  <5x: {np.mean(qe<=5)*100:.1f}%")
print(f"\n  GNN:          P50=1.57x  P90=4.51x  P95=9.74x  R²=0.923  <2x:62%  <5x:91%")
