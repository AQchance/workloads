"""
Run query_split_exp.py over multiple seeds and compile summary table.
Usage: python lstm/run_multi_seed.py --seeds 42,123,456,78,234,567
"""
import subprocess, sys, json, os

ROOT = '/home/anqian/Desktop/my_lab/workloads'
VENV = '/home/anqian/code/python/workloads/venv/bin/activate'
SCRIPT = 'lstm/query_split_exp.py'
OUT_DIR = os.path.join(ROOT, 'lstm', 'query_split_results')

if __name__ == '__main__':
    seeds = sys.argv[1:] if len(sys.argv) > 1 else ['42', '123', '456']
    results = {}

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"# SEED {seed}")
        print(f"{'#' * 60}")
        result_path = os.path.join(OUT_DIR, f'results_s{seed}.json')

        if os.path.exists(result_path):
            print(f"  Results already exist, loading...")
            with open(result_path) as f:
                results[seed] = json.load(f)
        else:
            cmd = f'cd {ROOT} && source {VENV} && PYTHONUNBUFFERED=1 python -u {SCRIPT} --seed {seed} --epochs 250'
            ret = subprocess.run(cmd, shell=True, executable='/bin/zsh')
            if ret.returncode != 0:
                print(f"  FAILED with code {ret.returncode}")
                continue
            with open(result_path) as f:
                results[seed] = json.load(f)

    # ─── Summary table ───
    print(f"\n{'=' * 80}")
    print("MULTI-SEED SUMMARY (Query-Level Split, 70/30 by Query ID)")
    print(f"{'=' * 80}")
    print(f"{'Seed':>6}  {'ICONQ P50':>10}  {'GNN P50':>8}  {'Δ_P50':>6}  "
          f"{'ICONQ P90':>10}  {'GNN P90':>8}  {'Δ_P90':>6}  "
          f"{'ICONQ P95':>10}  {'GNN P95':>8}  {'Δ_P95':>6}")
    print('-' * 88)

    for seed in seeds:
        if seed not in results:
            continue
        r = results[seed]
        ic = r['iconq']['metrics']
        gn = r['gnn']['metrics']
        d50 = (ic['P50'] - gn['P50']) / ic['P50'] * 100
        d90 = (ic['P90'] - gn['P90']) / ic['P90'] * 100
        d95 = (ic['P95'] - gn['P95']) / ic['P95'] * 100
        print(f'{seed:>6}  {ic["P50"]:>8.2f}x  {gn["P50"]:>8.2f}x  {d50:>+5.1f}%  '
              f'{ic["P90"]:>10.2f}x  {gn["P90"]:>8.2f}x  {d90:>+5.1f}%  '
              f'{ic["P95"]:>10.2f}x  {gn["P95"]:>8.2f}x  {d95:>+5.1f}%')

    # ─── Average over seeds ───
    print('\n' + '-' * 88)
    ic_p50s = [results[s]['iconq']['metrics']['P50'] for s in seeds if s in results]
    gn_p50s = [results[s]['gnn']['metrics']['P50'] for s in seeds if s in results]
    ic_p90s = [results[s]['iconq']['metrics']['P90'] for s in seeds if s in results]
    gn_p90s = [results[s]['gnn']['metrics']['P90'] for s in seeds if s in results]
    ic_p95s = [results[s]['iconq']['metrics']['P95'] for s in seeds if s in results]
    gn_p95s = [results[s]['gnn']['metrics']['P95'] for s in seeds if s in results]

    import numpy as np
    avg_ic50 = np.mean(ic_p50s); avg_gn50 = np.mean(gn_p50s)
    avg_ic90 = np.mean(ic_p90s); avg_gn90 = np.mean(gn_p90s)
    avg_ic95 = np.mean(ic_p95s); avg_gn95 = np.mean(gn_p95s)
    print(f'{"Mean":>6}  {avg_ic50:>8.2f}x  {avg_gn50:>8.2f}x  {(avg_ic50-avg_gn50)/avg_ic50*100:>+5.1f}%  '
          f'{avg_ic90:>10.2f}x  {avg_gn90:>8.2f}x  {(avg_ic90-avg_gn90)/avg_ic90*100:>+5.1f}%  '
          f'{avg_ic95:>10.2f}x  {avg_gn95:>8.2f}x  {(avg_ic95-avg_gn95)/avg_ic95*100:>+5.1f}%')
