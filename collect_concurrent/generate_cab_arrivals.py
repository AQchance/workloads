#!/usr/bin/env python3
"""
Generate query arrival times using CAB benchmark methodology.

CAB algorithm:
  1. Divide total_duration into 100 slots
  2. Assign each slot an intensity based on one of 5 arrival patterns
  3. Per slot: query_count = cpu_time_in_slot / avg_query_time
  4. Within slot: exponential inter-arrival times
  5. Stop when accumulated CPU time exceeds slot budget

Adapted for our 1000 SQLStorm queries with real cgroup execution times.
"""

import json, os, math, random, sys
import numpy as np

QUERY_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries_v2.txt'
CGROUP_DIR = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent'

# ─── CAB parameters ───
N_SLOTS = 100
TOTAL_DURATION_H = 2.0        # benchmark window in hours
TARGET_LOAD = 0.6             # fraction of system capacity to use (0-1)
PATTERN_ID = 1                # 1=multi-sine, 3=spikes, 4=burst, 5=constant+break, 6=hourly-spikes

SEED_DATABASE = 6
SEED_PATTERNS = 496
SEED_QUERIES = 8128


def load_query_runtimes():
    """Load actual execution times from cgroup data."""
    with open(QUERY_FILE) as f:
        qids = [l.strip() for l in f if l.strip()]

    runtimes = {}
    missing = 0
    for qid in qids:
        jf = os.path.join(CGROUP_DIR, f'{qid}.json')
        if os.path.exists(jf):
            with open(jf) as fp:
                d = json.load(fp)
            if d.get('status') == 'ok':
                runtimes[qid] = d['latency_s']
            else:
                runtimes[qid] = 60.0  # penalty → assume 60s
        else:
            runtimes[qid] = 30.0  # unknown → assume median
            missing += 1

    print(f"Loaded {len(runtimes)} queries ({missing} missing cgroup data, using defaults)")
    return qids, runtimes


def generate_deterministic_log_normal(n, mean_log, sd_log, seed):
    """CAB's deterministic log-normal: oversample, sort, evenly pick."""
    rng = random.Random(seed)
    oversample = n * 1000
    values = [rng.lognormvariate(mean_log, sd_log) for _ in range(oversample)]
    values.sort()
    step = oversample // n
    return [values[i * step] for i in range(n)]


def scale_to_target(values, target_sum):
    """Linearly scale values so their sum equals target_sum."""
    current_sum = sum(values)
    factor = target_sum / current_sum if current_sum > 0 else 1.0
    return [v * factor for v in values]


# ─── CAB's 5 arrival patterns (ported from benchmark.cpp) ───

def pattern_1_random_sines(rng):
    """Pattern 1: baseline + 8 random sine humps + noise + spikes."""
    slots = [0.2] * N_SLOTS  # constant baseline 0.2
    # Uniform random noise
    for i in range(N_SLOTS):
        slots[i] += rng.random()
    # 8 random sine humps
    for _ in range(8):
        center = rng.random()
        width = rng.uniform(0.05, 0.10)
        intensity = 0.5
        start = max(0.0, center - width / 2)
        end = min(1.0, center + width / 2)
        istart = int(start * N_SLOTS)
        iend = int(end * N_SLOTS)
        for i in range(istart, min(iend, N_SLOTS)):
            phase = (i / N_SLOTS - start) / max(width, 0.001)
            slots[i] += intensity * math.sin(phase * math.pi)
    # Random spike noise: 10% chance per slot
    for i in range(N_SLOTS):
        if rng.random() < 0.1:
            slots[i] += rng.random()
    return [max(0.0, s) for s in slots]


def pattern_3_random_spikes(rng):
    """Pattern 3: a few random walk bursts (1-5 spikes)."""
    slots = [0.0] * N_SLOTS
    n_spikes = round(rng.expovariate(1.0 / 0.5) * 2 + 1)
    n_spikes = max(1, min(n_spikes, 5))
    for _ in range(n_spikes):
        start_pct = rng.uniform(0.1, 0.9)
        length_pct = 0.10
        intensity = rng.random()
        istart = int(start_pct * N_SLOTS)
        iend = int(min(1.0, start_pct + length_pct) * N_SLOTS)
        # Random walk within spike region
        val = rng.random() * intensity
        for i in range(istart, min(iend, N_SLOTS)):
            val += rng.uniform(-0.3, 0.3) * intensity
            val = max(0.0, min(val, intensity))
            slots[i] = max(slots[i], val)
    return slots


def pattern_4_single_burst(rng):
    """Pattern 4: one short burst column (random walk)."""
    slots = [0.0] * N_SLOTS
    start_pct = rng.uniform(0.1, 0.9)
    length_pct = rng.uniform(0.15, 0.25)
    intensity = rng.random()
    istart = int(start_pct * N_SLOTS)
    iend = int(min(1.0, start_pct + length_pct) * N_SLOTS)
    val = rng.random() * intensity
    for i in range(istart, min(iend, N_SLOTS)):
        val += rng.uniform(-0.3, 0.3) * intensity
        val = max(0.0, min(val, intensity))
        slots[i] = val
    return slots


def pattern_5_constant_with_break(rng):
    """Pattern 5: constant load with optional sudden break/spikes."""
    slots = [4.0] * N_SLOTS
    # Optional positive spikes (50% each, up to 3)
    for _ in range(3):
        if rng.random() < 0.5:
            pos = int(rng.random() * N_SLOTS)
            slots[pos] += 6.0
    # Optional negative dip (50% chance)
    if rng.random() < 0.5:
        pos = int(rng.random() * N_SLOTS)
        slots[pos] = max(0, slots[pos] - 100.0)
    return [max(0.0, s) for s in slots]


def pattern_6_hourly_spikes(rng):
    """Pattern 6: spikes every hour (24 per timeline)."""
    slots = [0.0] * N_SLOTS
    level = rng.randint(0, 1)
    base = 2 * level
    slots_per_hour = N_SLOTS // 24
    for hour in range(24):
        istart = hour * slots_per_hour
        iend = min((hour + 1) * slots_per_hour, N_SLOTS)
        spike_width = rng.uniform(0.4, 0.6)
        spike_start = int(istart + (iend - istart) * (1 - spike_width) / 2)
        spike_end = int(istart + (iend - istart) * (1 + spike_width) / 2)
        intensity = rng.uniform(1.0, 6.0)
        for i in range(istart, iend):
            slots[i] = base
        for i in range(spike_start, min(spike_end, N_SLOTS)):
            slots[i] = intensity
    return slots


PATTERNS = {
    1: ('multi-sine', pattern_1_random_sines),
    3: ('random-spikes', pattern_3_random_spikes),
    4: ('single-burst', pattern_4_single_burst),
    5: ('constant-break', pattern_5_constant_with_break),
    6: ('hourly-spikes', pattern_6_hourly_spikes),
}


def generate_arrival_times(qids, runtimes, pattern_id, total_duration_h, target_load, seed):
    """Generate query arrival times using CAB methodology."""
    rng = random.Random(seed)

    # ─── Step 1: Generate arrival pattern (intensity per slot) ───
    pattern_rng = random.Random(SEED_PATTERNS)
    pattern_name, pattern_fn = PATTERNS[pattern_id]
    slots_intensity = pattern_fn(pattern_rng)
    # Normalize so sum = N_SLOTS (average intensity = 1.0)
    total_intensity = sum(slots_intensity)
    if total_intensity > 0:
        slots_intensity = [s * N_SLOTS / total_intensity for s in slots_intensity]

    # ─── Step 2: Calculate total CPU time needed ───
    total_cpu_seconds = sum(runtimes.values())
    # Apply target load: we want the system to be "target_load" utilized
    # Effective CPU = total_cpu_seconds * target_load distributed over total_duration_h
    # With K=2 workers, system capacity = 2 * total_duration_h * 3600 CPU-seconds
    # But we use target_load to control density directly
    avg_runtime = total_cpu_seconds / len(qids)
    print(f"\nQueries: {len(qids)} | Avg runtime: {avg_runtime:.1f}s | Total CPU: {total_cpu_seconds/3600:.1f} CPU-hours")

    # ─── Step 3: Distribute queries across slots, then generate arrivals ───
    total_duration_s = total_duration_h * 3600.0
    ms_per_slot = total_duration_s * 1000.0 / N_SLOTS
    slot_duration_s = ms_per_slot / 1000.0
    n_queries = len(qids)

    # Distribute query count across slots proportional to intensity
    # Each slot gets ceil(n * intensity_i / sum(intensity))
    total_intensity = sum(slots_intensity)
    queries_per_slot = []
    assigned = 0
    for slot in range(N_SLOTS):
        if slot == N_SLOTS - 1:
            n = n_queries - assigned  # all remaining
        else:
            n = max(0, int(n_queries * slots_intensity[slot] / total_intensity))
        queries_per_slot.append(n)
        assigned += n

    # Distribute any leftover queries to non-empty slots
    while assigned < n_queries:
        for slot in range(N_SLOTS):
            if assigned >= n_queries:
                break
            if slots_intensity[slot] > 0:
                queries_per_slot[slot] += 1
                assigned += 1

    # Generate arrival times within each slot
    all_queries = []
    query_idx = 0
    slot_seed_base = SEED_QUERIES

    for slot in range(N_SLOTS):
        n_in_slot = queries_per_slot[slot]
        if n_in_slot <= 0:
            continue

        slot_start_ms = slot * ms_per_slot
        slot_rng = random.Random(slot_seed_base + slot)

        # Exponential inter-arrival with minimum gap (shifted exponential)
        # Effective inter-arrival = MIN_GAP + Exp(rate), then scaled to fit slot
        MIN_GAP = 1.0  # seconds — minimum time between consecutive query arrivals
        if n_in_slot > 1:
            # Rate for the exponential part (after subtracting MIN_GAP per query)
            available_time = slot_duration_s - n_in_slot * MIN_GAP
            if available_time > 0:
                rate_per_second = (n_in_slot - 1) / available_time
            else:
                rate_per_second = (n_in_slot - 1) / 1.0  # very tight, fallback
            inter_arrivals = [max(0.01, slot_rng.expovariate(rate_per_second)) for _ in range(n_in_slot)]
            # Scale to fit exactly within slot (95% to avoid edge spill)
            total_inter = sum(inter_arrivals)
            scale = (available_time * 0.95) / max(total_inter, 0.001)
            now_s = 0.0
            for j in range(n_in_slot):
                if query_idx >= n_queries:
                    break
                qid = qids[query_idx]
                rt = runtimes.get(qid, avg_runtime)
                arrival_ms = slot_start_ms + now_s * 1000.0
                all_queries.append({
                    'query_idx': query_idx,
                    'qid': qid,
                    'arrival_time_s': round(arrival_ms / 1000.0, 3),
                    'estimated_runtime_s': round(rt, 2),
                })
                query_idx += 1
                if j < len(inter_arrivals):
                    now_s += inter_arrivals[j] * scale + MIN_GAP
        else:
            # Single query: place at slot center
            qid = qids[query_idx]
            rt = runtimes.get(qid, avg_runtime)
            arrival_ms = slot_start_ms + ms_per_slot / 2
            all_queries.append({
                'query_idx': query_idx,
                'qid': qid,
                'arrival_time_s': round(arrival_ms / 1000.0, 3),
                'estimated_runtime_s': round(rt, 2),
            })
            query_idx += 1

    # Sort by arrival time
    all_queries.sort(key=lambda x: x['arrival_time_s'])

    return all_queries, pattern_name, slots_intensity


def main():
    qids, runtimes = load_query_runtimes()
    avg_rt = np.mean(list(runtimes.values()))

    # K=2 system capacity: can process ~2 queries concurrently
    # With avg runtime ~30s, capacity ≈ 2 * 3600 / 30 = 240 queries per hour
    # 1000 queries at capacity would take ~4.2 hours
    # With target_load=0.6, total duration ~2h gives reasonable density
    K = 2
    capacity_qph = K * 3600 / avg_rt
    print(f"System: K={K} | Capacity: {capacity_qph:.0f} q/h")
    print(f"Target load: {TARGET_LOAD} | Duration: {TOTAL_DURATION_H}h")
    print(f"Pattern: {PATTERN_ID} ({PATTERNS[PATTERN_ID][0]})")

    arrivals, pattern_name, slots = generate_arrival_times(
        qids, runtimes, PATTERN_ID, TOTAL_DURATION_H, TARGET_LOAD, SEED_QUERIES)

    # ─── Summary ───
    print(f"\nGenerated {len(arrivals)} query arrivals")
    if arrivals:
        inter = np.diff([a['arrival_time_s'] for a in arrivals])
        inter_pos = inter[inter > 0.001]
        print(f"Duration: {arrivals[-1]['arrival_time_s']:.0f}s = {arrivals[-1]['arrival_time_s']/3600:.1f}h")
        print(f"Inter-arrival: median={np.median(inter_pos):.2f}s mean={np.mean(inter_pos):.2f}s")
        print(f"  P10={np.percentile(inter_pos,10):.1f}s P90={np.percentile(inter_pos,90):.1f}s P99={np.percentile(inter_pos,99):.1f}s")

        # Concurrency check: estimate peak concurrent queries
        # For each query, check how many others overlap with it
        starts = np.array([a['arrival_time_s'] for a in arrivals])
        ends = starts + np.array([a['estimated_runtime_s'] for a in arrivals])
        max_concurrent = 0
        for i in range(len(arrivals)):
            n_overlap = np.sum((starts >= starts[i]) & (starts < ends[i]))
            max_concurrent = max(max_concurrent, n_overlap)
        print(f"Peak concurrent queries (estimated): {max_concurrent}")

    # ─── Save ───
    out_path = os.path.join(OUT_DIR, 'arrival_times_cab.csv')
    with open(out_path, 'w') as f:
        f.write("query_order,qid,arrival_time_s,estimated_runtime_s\n")
        for a in arrivals:
            f.write(f"{a['query_idx']},{a['qid']},{a['arrival_time_s']},{a['estimated_runtime_s']}\n")
    print(f"Saved to {out_path}")

    # Also save the pattern for reference
    pattern_path = os.path.join(OUT_DIR, 'arrival_pattern_cab.json')
    with open(pattern_path, 'w') as f:
        json.dump({
            'pattern_id': PATTERN_ID,
            'pattern_name': pattern_name,
            'total_duration_h': TOTAL_DURATION_H,
            'target_load': TARGET_LOAD,
            'n_slots': N_SLOTS,
            'slots_intensity': slots,
        }, f, indent=2)
    print(f"Pattern saved to {pattern_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', type=int, default=1, choices=[1,3,4,5,6])
    p.add_argument('--duration', type=float, default=2.0)
    p.add_argument('--load', type=float, default=0.6)
    args = p.parse_args()

    PATTERN_ID = args.pattern
    TOTAL_DURATION_H = args.duration
    TARGET_LOAD = args.load

    main()
