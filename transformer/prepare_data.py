"""
Build Transformer training data from GNN features.

Each sample: a set of N concurrent queries.
  - node_features: [N, 134] per query (plan_emb + resources + serial_lat)
  - pair_features:  [N, N, 8] per pair (elapsed, is_before, overlap, conflict)
  - target: slowdown ratio for target query
"""

import os, json, csv, math, numpy as np

ROOT = '/home/anqian/Desktop/my_lab/workloads'
OUT = '/home/anqian/code/python/workloads/transformer'
GNF_FILES = [
    'lstm/gnn_features_k2_fixed.json',
    'lstm/gnn_features_k3_fixed.json',
    'lstm/gnn_features_k4_fixed.json',
]
TRACES = [
    'collect_concurrent/trace_2_mixed.csv',
    'collect_concurrent/trace_3_fixed_mixed.csv',
    'collect_concurrent/trace_4_fixed_mixed.csv',
]

MAX_NODES = 20  # max concurrent queries in a set


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_dataset():
    # Load GNN features
    gnf = {}
    for fn in GNF_FILES:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))

    # Build timeline
    timeline = []
    for tf in TRACES:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime']); st = row['status']
                actual = 60.0 if st == 'penalty' else rt
                timeline.append((float(row['start']), float(row['start']) + actual,
                                 row['qid'], actual, st))

    qid_info = {}
    for s, e, q, _, _ in timeline:
        if q not in qid_info: qid_info[q] = (s, e)

    # Build concurrent sets
    concurrent_sets = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        if qi not in gnf or sti == 'penalty': continue
        peers = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
                 if i != j and sj < ei and ej > si and qj in gnf]
        if peers:
            concurrent_sets.append((qi, si, ei, rti, peers))

    # Split by time
    mt = max(e for _, e, _, _, _ in timeline)
    sp = mt * 0.7
    train_sets = [cs for cs in concurrent_sets if cs[1] < sp]
    test_sets = [cs for cs in concurrent_sets if cs[1] >= sp]

    def build_features(sets):
        node_list, pair_list, time_list, res_list, target_list = [], [], [], [], []
        serial_lats = []

        for qi, si, ei, rti, peers in sets:
            all_queries = [qi] + peers[:MAX_NODES - 1]
            N = len(all_queries)
            if N < 2: continue

            # Node features
            nodes = np.zeros((MAX_NODES, 134), dtype=np.float32)
            times = np.zeros(MAX_NODES, dtype=np.float32)
            resources = np.zeros((MAX_NODES, 5), dtype=np.float32)
            for k, q in enumerate(all_queries):
                p = gnf[q]['plan_emb']
                r = list(gnf[q]['gpu_resources'].values())
                sl = gnf[q]['serial_labels'].get('latency_s', 1)
                nodes[k] = np.array(p + r + [math.log(1 + sl)], dtype=np.float32)
                times[k] = qid_info[q][0]
                resources[k] = np.array(r, dtype=np.float32)

            # Pair features: elapsed, is_before, overlap, resource_conflict
            pairs = np.zeros((MAX_NODES, MAX_NODES, 8), dtype=np.float32)
            for a in range(N):
                for b in range(N):
                    if a == b: continue
                    sa, ea = qid_info[all_queries[a]]
                    sb, eb = qid_info[all_queries[b]]
                    elapsed = sa - sb  # how long Q_a started after Q_b
                    is_before = 1.0 if sb < sa else 0.0
                    overlap_start = max(sa, sb)
                    overlap_end = min(ea, eb)
                    overlap = max(0.0, overlap_end - overlap_start) / max(ea - sa, 1.0)
                    conflict = resource_conflict(resources[a], resources[b])
                    pairs[a, b] = np.array([elapsed, is_before, overlap] + conflict,
                                           dtype=np.float32)

            # Target: slowdown ratio for Q_i (index 0 in nodes)
            serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
            target_ratio = rti / serial_lat

            node_list.append(nodes)
            pair_list.append(pairs)
            time_list.append(times)
            res_list.append(resources)
            target_list.append(target_ratio)
            serial_lats.append(serial_lat)

        return node_list, pair_list, time_list, res_list, target_list, serial_lats

    print('Building train...')
    tr = build_features(train_sets)
    print(f'  Train: {len(tr[0])} sets')
    print('Building test...')
    te = build_features(test_sets)
    print(f'  Test: {len(te[0])} sets')

    # Save
    for prefix, data in [('train', tr), ('test', te)]:
        node_arr = np.array(data[0], dtype=np.float32)
        pair_arr = np.array(data[1], dtype=np.float32)
        time_arr = np.array(data[2], dtype=np.float32)
        res_arr = np.array(data[3], dtype=np.float32)
        y_arr = np.array(data[4], dtype=np.float32)
        sl_arr = np.array(data[5], dtype=np.float32)

        # Normalize targets
        y_log = np.log(1 + y_arr)
        ym, ys = float(y_log.mean()), float(y_log.std()) + 1e-8
        y_norm = (y_log - ym) / ys

        np.savez(os.path.join(OUT, f'{prefix}_transformer.npz'),
                 nodes=node_arr, pairs=pair_arr, times=time_arr,
                 resources=res_arr, y=y_norm, y_mean=ym, y_std=ys,
                 serial_lat=sl_arr)
        print(f'  Saved {prefix}_transformer.npz ({len(y_arr)} sets)')

    return len(tr[0]), len(te[0])


if __name__ == '__main__':
    build_dataset()
