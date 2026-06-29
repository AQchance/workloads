"""
TabPFN v2 vs XGBoost: Latency Prediction Comparison.

TabPFN (Prior-Data Fitted Networks) is a pre-trained transformer for tabular data.
Unlike XGBoost, it requires NO hyperparameter tuning and NO training in the traditional
sense — it performs in-context learning (similar to how LLMs do few-shot learning).

Key modeling decisions for this domain:
  1. Features: Same 90-dim flat features as XGBoost baseline (fair comparison)
     - 88 stats (mean/max/sum/std for each of 22 node features)
     - n_nodes + n_edges
  2. Target: log(1 + latency_s) — same log-transform as XGBoost
     (TabPFN can handle skewed distributions better, but log-transform keeps comparison fair)
  3. Normalization: Test BOTH raw and standardized features
     (TabPFN was pre-trained on diverse distributions; theory says raw may work better)
  4. TabPFN fit_mode='low_memory' for GTX 1650 (4GB VRAM)
  5. 5-fold CV, same splits as XGBoost for head-to-head comparison

Usage:
    cd /home/anqian/Desktop/my_lab/workloads
    source /home/anqian/code/python/workloads/venv/bin/activate

    # Prerequisite: authenticate with HuggingFace (one-time)
    #   1. Visit https://huggingface.co/Prior-Labs/tabpfn_3 → Accept license
    #   2. Run: huggingface-cli login
    #
    #    Or set: export HF_TOKEN="hf_xxx"

    python gnn/tabpfn_comparison.py --seed 42
"""

import os, sys, math, argparse, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cgroup import load_cgroup_labels, extract_cpu_resource, normalize_labels
from train_ndv import load_ndv_cache, load_dist_cache

ROOT = '/home/anqian/Desktop/my_lab/workloads'

# ─── Feature Extraction (same as baseline_comparison.py) ───

def extract_flat_features(graph):
    """
    Extract 90-dim flat statistical features from a PyG graph.

    For each of the 22 node features → mean, max, sum, std (22×4=88)
    + n_nodes (1) + n_edges (1) = 90 features.

    This is the SAME feature set used by the XGBoost baseline.
    """
    x = graph.x.numpy()
    feats = []
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(x.shape[0])  # n_nodes
    feats.append(graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0)
    return np.array(feats, dtype=np.float64)


def load_all_data(ndv_cache, dist_cache):
    """Load all matched (graph, cgroup_label, qid) tuples."""
    from train_ndv import parse_plan

    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
    plan_dir = os.path.join(ROOT, 'explain_plans')

    graphs, labels_out, qids = [], [], []
    for qid, clab in cgroup_labels.items():
        pf = os.path.join(plan_dir, f'{qid}.txt')
        if not os.path.exists(pf):
            continue
        with open(pf) as f:
            plan_text = f.read()
        g = parse_plan(plan_text, ndv_cache, dist_cache)
        if g is None or g.x.shape[0] == 0:
            continue
        # CPU resource from EXPLAIN ANALYZE
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                clab['cpu_resource_s'] = extract_cpu_resource(f.read())
        graphs.append(g)
        labels_out.append(clab)
        qids.append(qid)
    return graphs, labels_out, qids


# ─── XGBoost Baseline ───

def train_eval_xgboost(X_train, y_train, X_test, y_test, seed=42):
    """Train XGBoost and return Q-error metrics."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=seed, subsample=0.8
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


# ─── TabPFN ───

def try_import_tabpfn():
    """Try importing TabPFN. Returns (TabPFNRegressor, error_msg)."""
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor, None
    except ImportError as e:
        return None, f"TabPFN not installed: {e}\n  Run: pip install tabpfn"
    except Exception as e:
        return None, str(e)


def train_eval_tabpfn(X_train, y_train, X_test, y_test, random_state=42,
                       fit_mode='low_memory', device='cuda'):
    """
    Train TabPFN via in-context learning.

    TabPFN doesn't do gradient descent — fit() is a forward pass that
    conditions the pre-trained transformer on the training data.

    Args:
        fit_mode: 'low_memory' for <8GB GPUs, 'fit_preprocessors' for >=8GB
        device: 'cuda' or 'cpu'
    """
    TabPFNRegressor, err = try_import_tabpfn()
    if TabPFNRegressor is None:
        raise RuntimeError(err)

    model = TabPFNRegressor(
        n_estimators=8,
        random_state=random_state,
        fit_mode=fit_mode,
        device=device,
        inference_precision='auto',
    )

    # TabPFN expects float32
    X_train_f32 = X_train.astype(np.float32)
    y_train_f32 = y_train.astype(np.float32)
    X_test_f32 = X_test.astype(np.float32)

    model.fit(X_train_f32, y_train_f32)
    preds = model.predict(X_test_f32)
    return preds.astype(np.float64)


# ─── Evaluation ───

def compute_metrics(y_true, y_pred, y_mean, y_std):
    """
    Compute Q-error metrics in original (non-log) space.

    Args:
        y_true, y_pred: z-scored log(1+x) values
        y_mean, y_std: normalization stats
    """
    # Denormalize to log(1+x) space
    log_true = y_true * y_std + y_mean
    log_pred = y_pred * y_std + y_mean

    # Convert to original space
    true_raw = np.exp(log_true) - 1
    pred_raw = np.exp(log_pred) - 1

    # Clamp to reasonable minimum
    true_raw = np.maximum(true_raw, 0.01)
    pred_raw = np.maximum(pred_raw, 0.01)

    # Q-error
    qe = np.maximum(pred_raw / true_raw, true_raw / pred_raw)
    qe_sorted = np.sort(qe)
    n = len(qe_sorted)

    # R² in log space
    r2 = 1 - np.sum((log_true - log_pred) ** 2) / max(
        np.sum((log_true - np.mean(log_true)) ** 2), 1e-8
    )

    return {
        'P50': qe_sorted[n // 2],
        'P90': qe_sorted[int(n * 0.9)],
        'P95': qe_sorted[int(n * 0.95)],
        'R2': r2,
        'within_2x': np.mean(qe <= 2.0),
    }


def print_metrics_table(name, metrics, dim_names):
    """Pretty-print a metrics table."""
    print(f"\n  {name}:")
    header = f"    {'Dim':<12} {'P50':>7} {'P90':>7} {'P95':>7} {'R²':>7} {'<2x':>7}"
    print(header)
    print("    " + "-" * 47)
    for dim in dim_names:
        m = metrics[dim]
        print(f"    {dim:<12} {m['P50']:.2f}x {m['P90']:.2f}x {m['P95']:.2f}x "
              f"{m['R2']:.3f} {m['within_2x']:.1%}")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fit-mode', default='low_memory',
                        choices=['low_memory', 'fit_preprocessors', 'fit_with_cache'])
    parser.add_argument('--skip-tabpfn', action='store_true',
                        help='Skip TabPFN if HF auth not set up')
    args = parser.parse_args()

    print("=" * 70)
    print("TabPFN v2 vs XGBoost: Latency Prediction Comparison")
    print(f"Seed={args.seed}, {args.n_folds}-Fold CV")
    print("=" * 70)

    # ─── Load Data ───
    print("\n[1/5] Loading data...")
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
    dist_cache = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))
    graphs, labels, qids = load_all_data(ndv_cache, dist_cache)
    print(f"  Matched: {len(graphs)} queries")

    # Extract flat features
    print("\n[2/5] Extracting 90-dim flat features...")
    X_raw = np.array([extract_flat_features(g) for g in graphs], dtype=np.float64)
    print(f"  Feature matrix: {X_raw.shape}")

    # Extract labels (log(1+x) transformed)
    y_all = np.array([
        math.log(1 + max(l['latency_s'], 0.001)) for l in labels
    ], dtype=np.float64)
    y_all_mem = np.array([
        math.log(1 + max(l['memory_bytes'], 1)) for l in labels
    ], dtype=np.float64)
    y_all_disk = np.array([
        math.log(1 + max(l['disk_bytes'], 1)) for l in labels
    ], dtype=np.float64)
    y_all_net = np.array([
        math.log(1 + max(l['network_bytes'], 1)) for l in labels
    ], dtype=np.float64)
    y_all_cpu = np.array([
        math.log(1 + max(l.get('cpu_resource_s', 0.001), 0.001)) for l in labels
    ], dtype=np.float64)

    dim_targets = {
        'Latency': y_all,
        'Memory': y_all_mem,
        'DiskIO': y_all_disk,
        'Network': y_all_net,
        'CPU_Res': y_all_cpu,
    }

    print(f"  Latency (log): [{y_all.min():.1f}, {y_all.mean():.1f}, {y_all.max():.1f}]")
    print(f"    → exp range: [{np.exp(y_all.min()):.1f}s, {np.exp(y_all.mean()):.1f}s, "
          f"{np.exp(y_all.max()):.1f}s]")

    # Filter latency <= 200s for cleaner comparison
    max_lat = 200.0
    valid_mask = np.array([l['latency_s'] <= max_lat for l in labels])
    n_excluded = (~valid_mask).sum()
    if n_excluded > 0:
        print(f"  Excluding {n_excluded} queries with latency > {max_lat}s")
        X_valid = X_raw[valid_mask]
        for k in dim_targets:
            dim_targets[k] = dim_targets[k][valid_mask]
    else:
        X_valid = X_raw

    n_queries = len(X_valid)
    print(f"  Final: {n_queries} queries")

    # ─── 5-Fold Split ───
    print(f"\n[3/5] Creating {args.n_folds}-fold splits...")
    np.random.seed(args.seed)
    indices = np.random.permutation(n_queries)
    fold_size = n_queries // args.n_folds
    folds = []
    for i in range(args.n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < args.n_folds - 1 else n_queries
        test_idx = indices[test_start:test_end]
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
        print(f"  Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    # ─── TabPFN Check ───
    TabPFNRegressor, tabpfn_err = try_import_tabpfn()
    tabpfn_available = TabPFNRegressor is not None
    if not tabpfn_available and not args.skip_tabpfn:
        print(f"\n[!] TabPFN not available: {tabpfn_err}")
        print("    To use TabPFN:")
        print("    1. Visit https://huggingface.co/Prior-Labs/tabpfn_3 → Accept license")
        print("    2. Run: huggingface-cli login")
        print("    Then re-run without --skip-tabpfn")
        print("\n    Continuing with XGBoost-only comparison...")

    # ─── Run Comparison ───
    print(f"\n[4/5] Running {args.n_folds}-Fold CV...")

    dim_names = ['Latency', 'Memory', 'DiskIO', 'Network', 'CPU_Res']

    # Storage for fold results
    xgb_results = {d: [] for d in dim_names}
    tabpfn_raw_results = {d: [] for d in dim_names}
    tabpfn_std_results = {d: [] for d in dim_names}

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  --- Fold {fold_i + 1}/{args.n_folds} ---")
        X_tr_raw = X_valid[train_idx]
        X_te_raw = X_valid[test_idx]

        # Standardize features (for XGBoost and TabPFN-std)
        Xm = X_tr_raw.mean(axis=0)
        Xs = X_tr_raw.std(axis=0) + 1e-8
        X_tr_std = (X_tr_raw - Xm) / Xs
        X_te_std = (X_te_raw - Xm) / Xs

        for dim in dim_names:
            y_all_dim = dim_targets[dim]
            y_tr = y_all_dim[train_idx]
            y_te = y_all_dim[test_idx]

            # Z-score normalize target (same for all models)
            ym = y_tr.mean()
            ys = max(y_tr.std(), 1e-8)
            y_tr_z = (y_tr - ym) / ys
            y_te_z = (y_te - ym) / ys

            # 1) XGBoost (standardized features)
            t0 = time.time()
            xgb_pred_z = train_eval_xgboost(X_tr_std, y_tr_z, X_te_std, y_te_z,
                                             seed=args.seed + fold_i)
            xgb_time = time.time() - t0
            xgb_results[dim].append(compute_metrics(y_te_z, xgb_pred_z, ym, ys))

            # 2) TabPFN on raw features (no standardization)
            if tabpfn_available and not args.skip_tabpfn:
                try:
                    t0 = time.time()
                    tabpfn_pred_z = train_eval_tabpfn(
                        X_tr_raw, y_tr_z, X_te_raw, y_te_z,
                        random_state=args.seed + fold_i,
                        fit_mode=args.fit_mode,
                        device=args.device,
                    )
                    tabpfn_time = time.time() - t0
                    tabpfn_raw_results[dim].append(
                        compute_metrics(y_te_z, tabpfn_pred_z, ym, ys)
                    )
                except Exception as e:
                    print(f"    [!] TabPFN (raw) error on {dim}: {e}")
                    tabpfn_raw_results[dim].append(None)

            # 3) TabPFN on standardized features
            if tabpfn_available and not args.skip_tabpfn:
                try:
                    t0 = time.time()
                    tabpfn_pred_z = train_eval_tabpfn(
                        X_tr_std, y_tr_z, X_te_std, y_te_z,
                        random_state=args.seed + fold_i,
                        fit_mode=args.fit_mode,
                        device=args.device,
                    )
                    tabpfn_time = time.time() - t0
                    tabpfn_std_results[dim].append(
                        compute_metrics(y_te_z, tabpfn_pred_z, ym, ys)
                    )
                except Exception as e:
                    print(f"    [!] TabPFN (std) error on {dim}: {e}")
                    tabpfn_std_results[dim].append(None)

        # Print per-fold latency result
        xgb_fold = xgb_results['Latency'][-1]
        print(f"    XGBoost            P50={xgb_fold['P50']:.2f}x  "
              f"P90={xgb_fold['P90']:.2f}x  R²={xgb_fold['R2']:.3f}  "
              f"({xgb_time:.1f}s)")
        if tabpfn_available and not args.skip_tabpfn:
            tf = tabpfn_raw_results['Latency'][-1]
            if tf:
                print(f"    TabPFN (raw)       P50={tf['P50']:.2f}x  "
                      f"P90={tf['P90']:.2f}x  R²={tf['R2']:.3f}  "
                      f"({tabpfn_time:.1f}s)")
            tf = tabpfn_std_results['Latency'][-1]
            if tf:
                print(f"    TabPFN (std)       P50={tf['P50']:.2f}x  "
                      f"P90={tf['P90']:.2f}x  R²={tf['R2']:.3f}")

    # ─── Aggregate Results ───
    print(f"\n[5/5] Aggregating {args.n_folds}-Fold CV Results")
    print("=" * 70)

    def aggregate(results_list):
        """Average metrics across folds, skipping None values."""
        valid = [r for r in results_list if r is not None]
        if not valid:
            return None
        agg = {}
        for key in valid[0]:
            agg[key] = np.mean([r[key] for r in valid])
        return agg

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({n_queries} queries, {args.n_folds}-Fold CV, seed={args.seed})")
    print(f"{'='*70}")

    for dim in dim_names:
        print(f"\n  {'─'*50}")
        print(f"  {dim}")
        print(f"  {'─'*50}")
        xgb_agg = aggregate(xgb_results[dim])
        print(f"    {'Model':<22} {'P50':>7} {'P90':>7} {'P95':>7} {'R²':>7} {'<2x':>7}")
        print(f"    {'-'*50}")
        print(f"    {'XGBoost (baseline)':<22} {xgb_agg['P50']:.2f}x "
              f"{xgb_agg['P90']:.2f}x {xgb_agg['P95']:.2f}x "
              f"{xgb_agg['R2']:.3f} {xgb_agg['within_2x']:.1%}")

        if tabpfn_available and not args.skip_tabpfn:
            tr_agg = aggregate(tabpfn_raw_results[dim])
            ts_agg = aggregate(tabpfn_std_results[dim])
            if tr_agg:
                delta_p50 = (tr_agg['P50'] - xgb_agg['P50']) / xgb_agg['P50'] * 100
                marker = "← BETTER" if delta_p50 < 0 else ""
                print(f"    {'TabPFN (raw feat)':<22} {tr_agg['P50']:.2f}x "
                      f"{tr_agg['P90']:.2f}x {tr_agg['P95']:.2f}x "
                      f"{tr_agg['R2']:.3f} {tr_agg['within_2x']:.1%}  "
                      f"ΔP50={delta_p50:+.1f}% {marker}")
            if ts_agg:
                delta_p50 = (ts_agg['P50'] - xgb_agg['P50']) / xgb_agg['P50'] * 100
                marker = "← BETTER" if delta_p50 < 0 else ""
                print(f"    {'TabPFN (std feat)':<22} {ts_agg['P50']:.2f}x "
                      f"{ts_agg['P90']:.2f}x {ts_agg['P95']:.2f}x "
                      f"{ts_agg['R2']:.3f} {ts_agg['within_2x']:.1%}  "
                      f"ΔP50={delta_p50:+.1f}% {marker}")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print("SUMMARY: TabPFN vs XGBoost")
    print(f"{'='*70}")
    print(f"  Data: {n_queries} queries, {X_valid.shape[1]} features")

    if tabpfn_available and not args.skip_tabpfn:
        print(f"\n  TabPFN works by in-context learning — no gradient descent.")
        print(f"  Pre-trained on millions of synthetic datasets, it conditions on")
        print(f"  your training data in a single forward pass.")
        print(f"\n  If TabPFN outperforms XGBoost, it demonstrates that the pre-training")
        print(f"  prior captures query execution plan patterns effectively.")
        print(f"\n  If XGBoost still wins, it confirms that 2306 samples with carefully")
        print(f"  engineered features is sufficient for gradient boosting to reach")
        print(f"  near-optimal performance for this task.")
        print(f"\n  In either case: the result is scientifically informative for your thesis.")
    else:
        print(f"\n  [!] TabPFN was not run. To enable it:")
        print(f"    Visit https://huggingface.co/Prior-Labs/tabpfn_3 → Accept license")
        print(f"    Run: huggingface-cli login")
        print(f"    Re-run without --skip-tabpfn")


if __name__ == '__main__':
    main()
