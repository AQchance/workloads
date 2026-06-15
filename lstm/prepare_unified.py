"""
Unified prepare script: produces BOTH original and completion-aware data
from the SAME train/test split. Guarantees fair comparison.
"""

import os, json, csv, math, numpy as np, sys

ROOT = '/home/anqian/Desktop/my_lab/workloads'
FEATURES = os.path.join(ROOT, 'lstm', 'gnn_features.json')
OUT = os.path.join(ROOT, 'lstm')
TRACES = [
    'collect_concurrent/trace_2.csv',
    'collect_concurrent/trace_3.csv',
    'collect_concurrent/trace_4.csv',
]

def load_iconq_features():
    """Merge ICONQ features from K2, K3, K4."""
    feats = {}
    for fn in ['lstm/iconq_features_v2.json','lstm/iconq_features_v2_k3.json','lstm/iconq_features_v2_k4.json']:
        with open(os.path.join(ROOT, fn)) as f:
            feats.update(json.load(f))
    return feats


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


with open(FEATURES) as f:
    gnf = json.load(f)

# Load and merge all traces
timeline = []
for tf in TRACES:
    with open(os.path.join(ROOT, tf)) as f:
        for row in csv.DictReader(f):
            rt = float(row['runtime']); st = row['status']
            actual = 60.0 if st == 'penalty' else rt
            s = float(row['start'])
            timeline.append((s, s + actual, row['qid'], actual, st))

# Build qid → (start, end) lookup
qid_info = {}
for s, e, q, _, _ in timeline:
    if q not in qid_info:
        qid_info[q] = (s, e)

# Build concurrent overlap lists (same for both versions)
ci = []
for i, (si, ei, qi, rti, sti) in enumerate(timeline):
    ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
          if i != j and sj < ei and ej > si]
    ci.append((qi, si, ei, rti, sti, ov))

# Split by time (accept ratio from command line, default 0.7)
split_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
split_prefix = sys.argv[2] if len(sys.argv) > 2 else ''
tag = f'_sr{int(split_ratio*100)}_{split_prefix}' if split_prefix else f'_sr{int(split_ratio*100)}'
mt = max(e for _, _, e, _, _, _ in ci)
split_time = mt * split_ratio
train_d = [c for c in ci if c[1] < split_time]
test_d = [c for c in ci if c[1] >= split_time]
print(f'Split ratio: {split_ratio}, tag: {tag}')
print(f'Train queries: {len(train_d)}, Test queries: {len(test_d)}')


def build_original(data_list):
    """Standard version: only submission events."""
    X, yr, sl = [], [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [math.log(1 + serial_lat)]
        t_res = list(gnf[qi]['gpu_resources'].values())
        seq = []
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in gnf and oq in qid_info]
        oi.sort()
        for os_val, oq in oi:
            oslv = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [oslv]
            c = resource_conflict(t_res, list(gnf[oq]['gpu_resources'].values()))
            feat = qv + ovv + [si - os_val, 1.0 if os_val < si else 0.0] + c
            seq.append(feat)
        if seq:
            X.append(seq); yr.append(rti / serial_lat); sl.append(serial_lat)
    return X, yr, sl


def build_completion(data_list):
    """Completion-aware: submission + completion events."""
    X, yr, sl = [], [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [math.log(1 + serial_lat)]
        t_res = list(gnf[qi]['gpu_resources'].values())
        events = []
        for oq in ov:
            if oq not in gnf or oq not in qid_info: continue
            os_val, oe_val = qid_info[oq]
            events.append((os_val, oq, 0))  # submission
            if oe_val < ei:
                events.append((oe_val, oq, 1))  # completion
        events.sort()
        seq = []
        for et, oq, etype in events:
            oslv = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [oslv]
            c = resource_conflict(t_res, list(gnf[oq]['gpu_resources'].values()))
            feat = qv + ovv + [si - et, 1.0 if et < si else 0.0] + c + [float(etype)]
            seq.append(feat)
        if seq:
            X.append(seq); yr.append(rti / serial_lat); sl.append(serial_lat)
    return X, yr, sl


def save_npz(X_seq, y_ratio, slats, ml, d, prefix):
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq):
        Xa[i, :len(s)] = s

    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_seq):
        mask[i, :len(s)] = 1.0
    Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

    yl = np.log(1 + np.array(y_ratio, dtype=np.float32))
    ym, ys_val = float(yl.mean()), float(yl.std()) + 1e-8

    np.savez(os.path.join(OUT, f'{prefix}.npz'),
             X=(Xa - Xm) / Xs,
             lengths=np.array([len(s) for s in X_seq], dtype=np.int32),
             y=(yl - ym) / ys_val,
             serial_lat=np.array(slats, dtype=np.float32),
             y_mean=ym, y_std=ys_val)
    print(f"  Saved {prefix}.npz ({len(X_seq)} seqs, dim={d})")


def build_iconq(data_list):
    """ICONQ-style flat vector features (no GNN)."""
    iconq_feats = load_iconq_features()
    X, yr = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in iconq_feats or sti == 'penalty': continue
        qv = iconq_feats[qi]['iconq_feat']
        seq = []
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in iconq_feats and oq in qid_info]
        oi.sort()
        for os_val, oq in oi:
            feat = qv + iconq_feats[oq]['iconq_feat'] + [si - os_val, 1.0 if os_val < si else 0.0]
            seq.append(feat)
        if seq:
            X.append(seq); yr.append(rti)
    return X, yr


def save_iconq_data(X_seq, y_abs, ml, d, prefix):
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
    Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    yl = np.log(1 + np.array(y_abs, dtype=np.float32))
    ym, ys_val = float(yl.mean()), float(yl.std()) + 1e-8
    np.savez(os.path.join(OUT, f'{prefix}.npz'),
             X=(Xa - Xm) / Xs, lengths=np.array([len(s) for s in X_seq], dtype=np.int32),
             y=(yl - ym) / ys_val, y_mean=ym, y_std=ys_val)
    print(f"  Saved {prefix}.npz ({len(X_seq)} seqs, dim={d})")


# ─── Build train for both versions ───
print("Building ORIGINAL train...")
X_tr_orig, y_tr_orig, sl_tr_orig = build_original(train_d)
print(f"  {len(X_tr_orig)} train seqs")

print("Building ORIGINAL test...")
X_te_orig, y_te_orig, sl_te_orig = build_original(test_d)
print(f"  {len(X_te_orig)} test seqs, median slowdown: {np.median(y_te_orig):.2f}")

ml_orig = max(max(len(s) for s in X_tr_orig), max(len(s) for s in X_te_orig))
d_orig = len(X_tr_orig[0][0])
print(f"  max_len={ml_orig}, dim={d_orig}")

# ─── Build completion-aware for both versions ───
print("Building COMPLETION train...")
X_tr_comp, y_tr_comp, sl_tr_comp = build_completion(train_d)
print(f"  {len(X_tr_comp)} train seqs")

print("Building COMPLETION test...")
X_te_comp, y_te_comp, sl_te_comp = build_completion(test_d)
print(f"  {len(X_te_comp)} test seqs, median slowdown: {np.median(y_te_comp):.2f}")

ml_comp = max(max(len(s) for s in X_tr_comp), max(len(s) for s in X_te_comp))
d_comp = len(X_tr_comp[0][0])
print(f"  max_len={ml_comp}, dim={d_comp}")

# ─── Verify test sets use same queries ───
print(f"\nTest set alignment check:")
print(f"  Original test queries: {len(X_te_orig)}, Completion test queries: {len(X_te_comp)}")
print(f"  Same count: {len(X_te_orig) == len(X_te_comp)}")
# Compare y_ratio (slowdown ratios, should be identical for same target queries)
corr = np.corrcoef(y_te_orig, y_te_comp)[0, 1]
print(f"  y_ratio correlation: {corr:.4f} (should be 1.0 if same queries)")

# ─── Save all three versions ───
print(f"\n=== Saving (split {split_ratio}, tag={tag}) ===")

# GNN original
print("GNN original...")
save_npz(X_tr_orig, y_tr_orig, sl_tr_orig, ml_orig, d_orig, f'train_gnn{tag}')
save_npz(X_te_orig, y_te_orig, sl_te_orig, ml_orig, d_orig, f'test_gnn{tag}')

# GNN completion-aware
print("GNN completion-aware...")
save_npz(X_tr_comp, y_tr_comp, sl_tr_comp, ml_comp, d_comp, f'train_comp{tag}')
save_npz(X_te_comp, y_te_comp, sl_te_comp, ml_comp, d_comp, f'test_comp{tag}')

# ICONQ
print("ICONQ...")
X_tr_iconq, y_tr_iconq = build_iconq(train_d)
X_te_iconq, y_te_iconq = build_iconq(test_d)
print(f"  Train: {len(X_tr_iconq)}, Test: {len(X_te_iconq)}")
ml_i = max(max(len(s) for s in X_tr_iconq), max(len(s) for s in X_te_iconq))
d_i = len(X_tr_iconq[0][0])
save_iconq_data(X_tr_iconq, y_tr_iconq, ml_i, d_i, f'train_iconq{tag}')
save_iconq_data(X_te_iconq, y_te_iconq, ml_i, d_i, f'test_iconq{tag}')

print(f"\nDone! All data saved with tag={tag}")
print(f"  GNN original:     train_gnn{tag}.npz / test_gnn{tag}.npz")
print(f"  GNN completion:   train_comp{tag}.npz / test_comp{tag}.npz")
print(f"  ICONQ:            train_iconq{tag}.npz / test_iconq{tag}.npz")
