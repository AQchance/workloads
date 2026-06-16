"""
Prepare data for Transformer V2 with dynamic batch masking and pair features.
"""

import os, json, csv, math, numpy as np
ROOT = '/home/anqian/Desktop/my_lab/workloads'
GNF_FILES = ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json', 'lstm/gnn_features_k4_fixed.json']
TRACES = ['collect_concurrent/trace_2_mixed.csv', 'collect_concurrent/trace_3_fixed_mixed.csv', 'collect_concurrent/trace_4_fixed_mixed.csv']


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_dataset():
    gnf = {}
    for fn in GNF_FILES:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))

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
        qid_info[q] = (s, e)

    # Build per-query features
    query_nodes = {}  # qid → [plan_emb(128) + resources(5) + serial_lat_log(1)]
    for qid in gnf:
        p = gnf[qid]['plan_emb']
        r = list(gnf[qid]['gpu_resources'].values())
        sl = gnf[qid]['serial_labels'].get('latency_s', 1)
        query_nodes[qid] = p + r + [math.log(1 + max(sl, 0.01))]

    # Build concurrent sets
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        if qi not in gnf or sti == 'penalty': continue
        peers = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
                 if i != j and sj < ei and ej > si and qj in gnf]
        if peers: ci.append((qi, si, ei, rti, peers))

    # Split by time
    mt = max(e for _, e, _, _, _ in timeline)
    sp = mt * 0.7
    train_sets = [cs for cs in ci if cs[1] < sp]
    test_sets = [cs for cs in ci if cs[1] >= sp]

    def build_features(sets):
        nodes_list, pairs_list, times_list, mask_list, target_list = [], [], [], [], []

        for qi, si, ei, rti, peers in sets:
            all_q = [qi] + peers
            N = len(all_q)

            # Node features
            nodes = np.zeros((N, 134), dtype=np.float32)
            times = np.zeros(N, dtype=np.float32)
            for k, q in enumerate(all_q):
                nodes[k] = np.array(query_nodes[q], dtype=np.float32)
                times[k] = qid_info[q][0]

            # Pair features: elapsed + is_before + overlap + 5-conflict = 8
            pairs = np.zeros((N, N, 8), dtype=np.float32)
            for a in range(N):
                for b in range(N):
                    if a == b: continue
                    sa, ea = qid_info[all_q[a]]
                    sb, eb = qid_info[all_q[b]]
                    elapsed = sa - sb
                    is_before = 1.0 if sb < sa else 0.0
                    overlap_start = max(sa, sb)
                    overlap_end = min(ea, eb)
                    overlap = max(0.0, overlap_end - overlap_start) / max(ea - sa, 1.0)
                    conflict = resource_conflict(nodes[a, 128:133], nodes[b, 128:133])
                    pairs[a, b] = np.array([elapsed, is_before, overlap] + conflict, dtype=np.float32)

            # Padding mask (all real queries, no padding in batch)
            mask = np.zeros(N, dtype=bool)

            # Target: slowdown ratio for Q_i (index 0)
            serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
            target_ratio = rti / serial_lat

            nodes_list.append(nodes)
            pairs_list.append(pairs)
            times_list.append(times)
            mask_list.append(mask)
            target_list.append(target_ratio)

        return nodes_list, pairs_list, times_list, mask_list, target_list

    print('Building train...')
    tr = build_features(train_sets)
    print(f'  Train: {len(tr[0])} sets')
    print('Building test...')
    te = build_features(test_sets)
    print(f'  Test: {len(te[0])} sets')

    # Save with padding to max N
    OUT = '/home/anqian/code/python/workloads/transformer'
    for prefix, data in [('train', tr), ('test', te)]:
        nodes_raw, pairs_raw, times_raw, masks_raw = data[0], data[1], data[2], data[3]
        y_arr = np.array(data[4], dtype=np.float32)
        y_log = np.log(1 + y_arr)
        ym, ys = float(y_log.mean()), float(y_log.std()) + 1e-8
        y_norm = (y_log - ym) / ys

        # Find max N across the dataset
        max_n = max(n.shape[0] for n in nodes_raw)
        n_samples = len(nodes_raw)
        print(f'  {prefix}: {n_samples} samples, max_n={max_n}')

        # Pad to max_n
        nodes_pad = np.zeros((n_samples, max_n, 134), dtype=np.float32)
        pairs_pad = np.zeros((n_samples, max_n, max_n, 8), dtype=np.float32)
        times_pad = np.zeros((n_samples, max_n), dtype=np.float32)
        masks_pad = np.ones((n_samples, max_n), dtype=bool)  # True = padding

        for i in range(n_samples):
            nn = nodes_raw[i].shape[0]
            nodes_pad[i, :nn] = nodes_raw[i]
            pairs_pad[i, :nn, :nn] = pairs_raw[i]
            times_pad[i, :nn] = times_raw[i]
            masks_pad[i, :nn] = False  # not padding

        np.savez(os.path.join(OUT, f'{prefix}_v2.npz'),
                 nodes=nodes_pad, pairs=pairs_pad, times=times_pad,
                 masks=masks_pad, y=y_norm, y_mean=ym, y_std=ys)
        print(f'  Saved {prefix}_v2.npz ({n_samples} sets)')

    return len(tr[0]), len(te[0])


if __name__ == '__main__':
    build_dataset()
