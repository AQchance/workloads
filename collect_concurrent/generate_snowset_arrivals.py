#!/usr/bin/env python3
"""
Generate arrival times by scaling Snowset real production traces to match
our query workload's CPU capacity.

Approach:
  1. Load Snowset arrival traces from ICONQ project
  2. Extract empirical inter-arrival time distribution
  3. Scale by factor so total CPU load fits our queries
  4. Sample inter-arrival times and generate cumulative arrivals
"""

import json, os, random, csv
import numpy as np

QUERY_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries_v2.txt'
CGROUP_DIR = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
SNOWSET_DIR = '/home/anqian/Desktop/python/IconqSched/workloads/snowset'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent'

SEED = 42


def load_runtimes():
    with open(QUERY_FILE) as f:
        qids = [l.strip() for l in f if l.strip()]
    runtimes = {}
    for qid in qids:
        jf = os.path.join(CGROUP_DIR, f'{qid}.json')
        if os.path.exists(jf):
            with open(jf) as fp:
                d = json.load(fp)
            runtimes[qid] = d['latency_s'] if d.get('status') == 'ok' else 60.0
        else:
            runtimes[qid] = 30.0
    return qids, runtimes


def load_snowset_inter_arrivals():
    """Extract inter-arrival times from ICONQ's Snowset traces."""
    import pandas as pd

    all_inter = []
    for fname in ['snowset_1399565385033429562', 'snowset_1453912639619907921']:
        path = os.path.join(SNOWSET_DIR, f'{fname}.csv')
        df = pd.read_csv(path)
        if 'g_offset_since_start_s' not in df.columns:
            # Needs preprocessing: createdTime → compute offset
            df['createdTime'] = pd.to_datetime(df['createdTime'], format='mixed')
            df = df.sort_values(by=['createdTime'], ascending=True)
            df['timestamp_s'] = df['createdTime'].astype('int64') / 1e9
            df['g_offset_since_start_s'] = df['timestamp_s'] - df['timestamp_s'].min()
            df['run_time_s'] = df['durationTotal'] / 1000

        offsets = df['g_offset_since_start_s'].sort_values().values
        inter = np.diff(offsets)
        inter = inter[(inter > 0.01) & (inter < 3600)]  # filter unreasonable gaps
        all_inter.append(inter)

    combined = np.concatenate(all_inter)
    print(f"Loaded {len(combined)} inter-arrival samples from Snowset")
    print(f"  Raw: median={np.median(combined):.2f}s mean={combined.mean():.2f}s")
    return combined


def main():
    qids, runtimes = load_runtimes()
    avg_rt = np.mean(list(runtimes.values()))
    total_cpu_s = sum(runtimes.values())
    n_queries = len(qids)

    # Load Snowset inter-arrival distribution
    snowset_inter = load_snowset_inter_arrivals()

    # ── Compute scaling factor ──
    # With target load ρ, we need total duration D such that:
    #   total_cpu_s / D ≈ ρ  (for single-server equivalent)
    # For c concurrent servers: total_cpu_s / (c * D) ≈ ρ
    # We use c=1 here because we want the ARRIVAL pattern, not execution capacity
    # The arrival rate λ = n / D
    # The service rate μ = 1 / avg_rt (per server)
    # With c servers: ρ = λ / (c * μ) = (n/D) * avg_rt / c
    # So D = n * avg_rt / (c * ρ)
    # But c goes away since we're defining the workload, not the system
    # We want: mean inter-arrival × system_capacity_factor = balanced load
    # Simpler: target mean inter-arrival = avg_rt / TARGET_LOAD
    # With avg_rt=25.5s and TARGET_LOAD=0.6: mean inter-arrival = 25.5/0.6 = 42.5s

    target_mean_inter_arrival = avg_rt / TARGET_LOAD
    snowset_mean = snowset_inter.mean()
    scale_factor = target_mean_inter_arrival / snowset_mean

    print(f"\nScaling:")
    print(f"  Avg query runtime: {avg_rt:.1f}s")
    print(f"  Target load: {TARGET_LOAD}")
    print(f"  Target mean inter-arrival: {target_mean_inter_arrival:.1f}s")
    print(f"  Snowset mean: {snowset_mean:.2f}s")
    print(f"  Scale factor: {scale_factor:.1f}x")

    # ── Generate arrivals ──
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    # Take a CONTIGUOUS segment of the Snowset trace to preserve burst structure
    # Scale the inter-arrival times, then extract n_inter consecutive values
    n_inter = n_queries - 1
    scaled_inter = snowset_inter * scale_factor

    # Pick a random starting point in the scaled trace
    max_start = len(scaled_inter) - n_inter - 1
    start_idx = np_rng.randint(0, max_start)
    sampled_inter = scaled_inter[start_idx:start_idx + n_inter]

    # Generate cumulative arrival times
    cumulative = np.zeros(n_queries)
    cumulative[1:] = np.cumsum(sampled_inter)
    total_duration_s = cumulative[-1]
    total_duration_h = total_duration_s / 3600

    # ── Build arrival records (queries in file order) ──
    arrivals = []
    for i, qid in enumerate(qids):
        arrivals.append({
            'query_idx': i,
            'qid': qid,
            'arrival_time_s': round(cumulative[i], 3),
            'estimated_runtime_s': round(runtimes.get(qid, avg_rt), 2),
        })

    # ── Concurrency stats ──
    starts = cumulative
    ends = starts + np.array([runtimes.get(q, avg_rt) for q in qids])
    samples = []
    for t in np.arange(0, total_duration_s, 10):
        n = np.sum((starts <= t) & (ends > t))
        samples.append(n)
    samples = np.array(samples)

    print(f"\nGenerated {len(arrivals)} arrivals over {total_duration_h:.1f}h")
    inter_all = sampled_inter
    print(f"Inter-arrival: median={np.median(inter_all):.2f}s mean={inter_all.mean():.2f}s")
    print(f"  P10={np.percentile(inter_all,10):.1f}s P25={np.percentile(inter_all,25):.1f}s")
    print(f"  P75={np.percentile(inter_all,75):.1f}s P90={np.percentile(inter_all,90):.1f}s P99={np.percentile(inter_all,99):.1f}s")

    print(f"\nConcurrency (sampled every 10s):")
    print(f"  Mean:{samples.mean():.1f}  Median:{np.median(samples):.1f}  P90:{np.percentile(samples,90):.1f}")
    print(f"  P95:{np.percentile(samples,95):.1f}  P99:{np.percentile(samples,99):.1f}  Max:{samples.max():.0f}")
    print(f"  % idle: {np.mean(samples==0)*100:.0f}%  % ≤2: {np.mean(samples<=2)*100:.0f}%  % ≤5: {np.mean(samples<=5)*100:.0f}%  % ≤10: {np.mean(samples<=10)*100:.0f}%")

    # ── Save ──
    out_path = os.path.join(OUT_DIR, 'arrival_times_cab.csv')
    with open(out_path, 'w') as f:
        f.write("query_order,qid,arrival_time_s,estimated_runtime_s\n")
        for a in arrivals:
            f.write(f"{a['query_idx']},{a['qid']},{a['arrival_time_s']},{a['estimated_runtime_s']}\n")
    print(f"\nSaved to {out_path}")

    # Show first arrivals
    print("\nFirst 15 arrivals:")
    for a in arrivals[:15]:
        print(f"  t={a['arrival_time_s']:8.1f}s  Q{a['qid']}  rt={a['estimated_runtime_s']:.0f}s")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--load', type=float, default=0.6, help='Target utilization (0.5-1.5)')
    args = p.parse_args()
    global TARGET_LOAD
    TARGET_LOAD = args.load
    main()
