"""
Step 2: Build concurrent training dataset from trace and GNN features.

For each query execution in the trace:
  - Find overlapping queries (concurrently executing)
  - Build interaction feature vectors (matching ICONQ format)
  - Split by time: first 70% train, last 30% test

Output: lstm/train_data.npz, lstm/test_data.npz
"""

import sys, os, json, csv, math
import numpy as np

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
TRACE_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'

PENALTY_ACTUAL_S = 60  # real wall-clock time for a TiDB restart cycle

# Load GNN features
with open(FEATURES_FILE) as f:
    gnn_features = json.load(f)
print(f"Loaded GNN features for {len(gnn_features)} queries")

# Load trace
trace = []
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f):
        trace.append(row)
print(f"Loaded {len(trace)} trace entries")

# Build execution timeline: [(start_s, end_s, qid, runtime, status)]
timeline = []
t = 0.0  # cumulative time
for r in trace:
    qid = r['qid']
    runtime = float(r['runtime'])
    status = r['status']

    if status == 'penalty':
        # Penalty: actual restart time is ~60s, not 600s
        actual_runtime = PENALTY_ACTUAL_S
    else:
        actual_runtime = runtime

    # Start time from trace is relative to script start. But the script
    # was restarted multiple times, so we need to detect timeline breaks.
    # Use start field + cumulative offset.
    start_in_batch = float(r['start'])
    if len(timeline) == 0:
        timeline.append((start_in_batch, start_in_batch + actual_runtime,
                         qid, actual_runtime, status))
    else:
        timeline.append((start_in_batch, start_in_batch + actual_runtime,
                         qid, actual_runtime, status))

# Compute concurrent overlaps: for each query, find which other queries
# have [start, end] overlapping with its [start, end]
print("Computing concurrent overlaps...")
concurrent_info = []  # list of (qid, start, end, runtime, status, [overlapping_qids])

for i, (start_i, end_i, qid_i, rt_i, st_i) in enumerate(timeline):
    overlaps = []
    for j, (start_j, end_j, qid_j, rt_j, st_j) in enumerate(timeline):
        if i == j:
            continue
        # Overlap: execution windows intersect
        if start_j < end_i and end_j > start_i:
            overlaps.append(qid_j)
    concurrent_info.append((qid_i, start_i, end_i, rt_i, st_i, overlaps))

# Split by time: first 70% → train, last 30% → test
max_time = max(e for _, _, e, _, _, _ in concurrent_info)
split_time = max_time * 0.7

train_data = [c for c in concurrent_info if c[1] < split_time]  # start time < split
test_data = [c for c in concurrent_info if c[1] >= split_time]

print(f"Train: {len(train_data)} executions")
print(f"Test:  {len(test_data)} executions")

# Build feature vectors for each execution
# Format: for each target query, build a list of interaction features
# Each interaction feature = [gnn_vec(target), gnn_vec(concurrent), time_diff, flags]

def build_dataset(concurrent_list):
    X = []   # list of lists (variable-length sequences)
    y = []   # runtimes
    meta = []  # (qid, n_concurrent) for debugging

    for qid, start, end, runtime, status, overlaps in concurrent_list:
        if qid not in gnn_features:
            continue

        serial_lat = math.log(1 + gnn_features[qid]['serial_labels'].get('latency_s', 10))
        target_vec = gnn_features[qid]['plan_emb'] + list(
            gnn_features[qid]['gpu_resources'].values()) + [serial_lat]
        # 128 plan_emb + 5 resources + 1 serial_lat = 134 dims per query

        seq = []
        # Sort concurrent queries by their start time
        overlap_info = []
        for oqid in overlaps:
            if oqid not in gnn_features:
                continue
            # Find this overlapping query's timing in the timeline
            for o_start, o_end, _, _, _ in [
                (s, e, _, _, _) for s, e, q, _, _ in timeline if q == oqid
            ]:
                overlap_info.append((o_start, oqid))
                break

        overlap_info.sort()

        for o_start, oqid in overlap_info:
            o_serial_lat = math.log(1 + gnn_features[oqid]['serial_labels'].get('latency_s', 10))
            ovec = gnn_features[oqid]['plan_emb'] + list(
                gnn_features[oqid]['gpu_resources'].values()) + [o_serial_lat]
            # Interaction feature: [target_vec, concurrent_vec, time_diff, is_before]
            time_diff = start - o_start  # negative if oqid started before target
            is_before = 1.0 if o_start < start else 0.0
            feat = target_vec + ovec + [time_diff, is_before]
            seq.append(feat)

        # Add the target query itself
        if len(seq) > 0:
            X.append(seq)
            y.append(runtime)
            meta.append((qid, len(overlaps)))

    return X, y, meta

print("\nBuilding training features...")
X_train, y_train, meta_train = build_dataset(train_data)
print(f"Train sequences: {len(X_train)}")

print("Building test features...")
X_test, y_test, meta_test = build_dataset(test_data)
print(f"Test sequences: {len(X_test)}")

# Save as numpy arrays with padding
# We'll use a simple format: each sequence padded to max length
# with metadata about original length

def pad_and_save(X, y, meta, prefix):
    max_len = max(len(seq) for seq in X)
    feat_dim = len(X[0][0])
    print(f"  {prefix}: max_seq_len={max_len}, feat_dim={feat_dim}")

    padded = np.zeros((len(X), max_len, feat_dim), dtype=np.float32)
    lengths = np.zeros(len(X), dtype=np.int32)
    labels = np.array(y, dtype=np.float32)

    for i, seq in enumerate(X):
        padded[i, :len(seq)] = seq
        lengths[i] = len(seq)

    out_path = os.path.join(OUT_DIR, f'{prefix}_data.npz')
    np.savez(out_path, X=padded, lengths=lengths, y=labels)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

pad_and_save(X_train, y_train, meta_train, 'train')
pad_and_save(X_test, y_test, meta_test, 'test')

# Stats
print(f"\n=== Dataset stats ===")
for name, X_set, y_set in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
    seq_lens = [len(s) for s in X_set]
    rt = np.array(y_set)
    print(f"  {name}: {len(X_set)} samples, "
          f"seq_len mean={np.mean(seq_lens):.1f} max={max(seq_lens)}, "
          f"runtime median={np.median(rt):.1f}s mean={np.mean(rt):.1f}s")
