#!/usr/bin/env python3
"""
Generate arrival times with periodic spike pattern:
  - Long quiet periods with very few queries
  - Periodic small spikes where concurrency reaches ~10
  - Spike every 15 minutes, lasting ~60 seconds, 25-30 queries per spike

Algorithm: divide timeline into cycles. Each cycle = quiet period + spike.
Within spike: tight exponential inter-arrival. Between spikes: sparse arrivals.
"""

import json, os, math, random
import numpy as np

QUERY_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries_v2.txt'
CGROUP_DIR = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent'

TOTAL_DURATION_H = 6.0
SPIKE_INTERVAL_MIN = 15       # spike every 15 minutes
SPIKE_DURATION_S = 60         # each spike lasts ~60 seconds
SPIKE_QUERIES = 22            # queries per spike (target concurrency ~10)
QUIET_QUERIES_PER_HOUR = 15   # queries trickling in during quiet periods
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


def main():
    qids, runtimes = load_runtimes()
    avg_rt = np.mean(list(runtimes.values()))
    total_duration_s = TOTAL_DURATION_H * 3600

    n_cycles = int(total_duration_s / (SPIKE_INTERVAL_MIN * 60))
    quiet_duration_s = SPIKE_INTERVAL_MIN * 60 - SPIKE_DURATION_S
    total_quiet_queries = int(TOTAL_DURATION_H * QUIET_QUERIES_PER_HOUR)
    total_spike_queries = n_cycles * SPIKE_QUERIES
    total_expected = total_spike_queries + total_quiet_queries
    remaining = len(qids) - total_expected

    # Adjust spike queries to fit exactly 1000
    spike_extra = remaining // n_cycles
    spike_queries_per = SPIKE_QUERIES + spike_extra
    extra_remaining = remaining - spike_extra * n_cycles

    print(f"Total: {len(qids)} queries, avg runtime: {avg_rt:.1f}s")
    print(f"Duration: {TOTAL_DURATION_H}h, cycles: {n_cycles} (every {SPIKE_INTERVAL_MIN}min)")
    print(f"Per spike: ~{spike_queries_per} queries in {SPIKE_DURATION_S}s window")
    print(f"Quiet: ~{total_quiet_queries + extra_remaining} queries, {QUIET_QUERIES_PER_HOUR}/h")
    print(f"Expected peak concurrency during spike: ~{spike_queries_per * avg_rt / SPIKE_DURATION_S:.0f}")

    rng = random.Random(SEED)
    all_arrivals = []
    query_idx = 0

    for cycle in range(n_cycles):
        cycle_start = cycle * SPIKE_INTERVAL_MIN * 60

        # ── Quiet phase ──
        n_quiet = total_quiet_queries // n_cycles
        if cycle < extra_remaining:
            n_quiet += 1
        if n_quiet > 0:
            # Evenly spread across quiet period
            gap = quiet_duration_s / (n_quiet + 1)
            for j in range(n_quiet):
                if query_idx >= len(qids):
                    break
                t = cycle_start + gap * (j + 1) + rng.uniform(-gap * 0.3, gap * 0.3)
                t = max(cycle_start, min(t, cycle_start + quiet_duration_s))
                qid = qids[query_idx]
                rt = runtimes.get(qid, avg_rt)
                all_arrivals.append({
                    'query_idx': query_idx,
                    'qid': qid,
                    'arrival_time_s': round(t, 3),
                    'estimated_runtime_s': round(rt, 2),
                })
                query_idx += 1

        # ── Spike phase ──
        spike_start = cycle_start + quiet_duration_s
        n_spike = spike_queries_per
        if cycle == n_cycles - 1:
            # Last cycle: dump all remaining queries
            n_spike = len(qids) - query_idx

        if n_spike > 0:
            # Tight exponential inter-arrival within spike window
            rate = (n_spike - 1) / SPIKE_DURATION_S if n_spike > 1 else 0.1
            now = 0.0
            for j in range(n_spike):
                if query_idx >= len(qids):
                    break
                qid = qids[query_idx]
                rt = runtimes.get(qid, avg_rt)
                t = spike_start + now
                all_arrivals.append({
                    'query_idx': query_idx,
                    'qid': qid,
                    'arrival_time_s': round(t, 3),
                    'estimated_runtime_s': round(rt, 2),
                })
                query_idx += 1
                gap = rng.expovariate(rate) if rate > 0 else SPIKE_DURATION_S
                now += max(1.0, gap)  # minimum 1s gap
                if now > SPIKE_DURATION_S * 1.5:  # allow slight overflow
                    break

    print(f"\nGenerated {len(all_arrivals)} arrivals")

    # ── Concurrency stats ──
    starts = np.array([a['arrival_time_s'] for a in all_arrivals])
    ends = starts + np.array([a['estimated_runtime_s'] for a in all_arrivals])
    max_t = ends.max()
    samples = []
    for t in np.arange(0, max_t, 10):
        n = np.sum((starts <= t) & (ends > t))
        samples.append(n)
    samples = np.array(samples)

    print(f"Concurrency (sampled every 10s):")
    print(f"  Mean:{samples.mean():.1f}  Median:{np.median(samples):.1f}  P90:{np.percentile(samples,90):.1f}")
    print(f"  P95:{np.percentile(samples,95):.1f}  P99:{np.percentile(samples,99):.1f}  Max:{samples.max():.0f}")
    print(f"  % idle: {np.mean(samples==0)*100:.0f}%  % ≤2: {np.mean(samples<=2)*100:.0f}%  % ≤5: {np.mean(samples<=5)*100:.0f}%")

    # ── Inter-arrival stats ──
    inter = np.diff(starts)
    inter_pos = inter[inter > 0.01]
    print(f"\nInter-arrival (all queries):")
    print(f"  Median:{np.median(inter_pos):.2f}s  Mean:{np.mean(inter_pos):.2f}s")
    print(f"  P10:{np.percentile(inter_pos,10):.1f}s  P25:{np.percentile(inter_pos,25):.1f}s")
    print(f"  P75:{np.percentile(inter_pos,75):.1f}s  P90:{np.percentile(inter_pos,90):.1f}s")

    # ── Save ──
    out_path = os.path.join(OUT_DIR, 'arrival_times_cab.csv')
    with open(out_path, 'w') as f:
        f.write("query_order,qid,arrival_time_s,estimated_runtime_s\n")
        for a in all_arrivals:
            f.write(f"{a['query_idx']},{a['qid']},{a['arrival_time_s']},{a['estimated_runtime_s']}\n")
    print(f"\nSaved to {out_path}")

    # Show first 2 spike cycles
    cycle_s = SPIKE_INTERVAL_MIN * 60
    for c in range(min(3, n_cycles)):
        in_cycle = [a for a in all_arrivals if c * cycle_s <= a['arrival_time_s'] < (c + 1) * cycle_s]
        spike_t = (c * cycle_s) + quiet_duration_s
        in_spike = [a for a in in_cycle if spike_t <= a['arrival_time_s'] < spike_t + SPIKE_DURATION_S * 1.5]
        print(f"\nCycle {c}: {len(in_cycle)} queries (spike: {len(in_spike)})")


if __name__ == '__main__':
    main()
