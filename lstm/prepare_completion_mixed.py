"""
Completion-aware mixed K=2+3+4 training data (optimized).
Pre-builds a qid → (start, end) lookup to avoid O(N^2) scanning.
"""

import os, json, csv, math, numpy as np

ROOT = '/home/anqian/Desktop/my_lab/workloads'
FEATURES = os.path.join(ROOT, 'lstm', 'gnn_features.json')
OUT = os.path.join(ROOT, 'lstm')
TRACES = ['collect_concurrent/trace_2.csv', 'collect_concurrent/trace_3.csv', 'collect_concurrent/trace_4.csv']


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


# Load features
with open(FEATURES) as f:
    gnf = json.load(f)

# Load and merge traces
timeline = []
for tf in TRACES:
    with open(os.path.join(ROOT, tf)) as f:
        for row in csv.DictReader(f):
            rt = float(row['runtime']); st = row['status']
            actual = 60.0 if st == 'penalty' else rt
            s = float(row['start'])
            timeline.append((s, s + actual, row['qid'], actual, st))

# Build fast lookup: qid → (start, end)
qid_info = {}
for s, e, q, _, _ in timeline:
    if q not in qid_info:
        qid_info[q] = (s, e)

# Build concurrent overlap lists
ci = []
for i, (si, ei, qi, rti, sti) in enumerate(timeline):
    ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si]
    ci.append((qi, si, ei, rti, sti, ov))

mt = max(e for _, _, e, _, _, _ in ci)
split_time = mt * 0.7
train_d = [c for c in ci if c[1] < split_time]
test_d = [c for c in ci if c[1] >= split_time]


def build_seq(data_list):
    X, yr, sl = [], [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty':
            continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        slv = math.log(1 + serial_lat)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [slv]
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
            X.append(seq)
            yr.append(rti / serial_lat)
            sl.append(serial_lat)
    return X, yr, sl


print('Building train...')
X_tr, y_tr, sl_tr = build_seq(train_d)
print(f'  Train: {len(X_tr)} seqs')
print('Building test...')
X_te, y_te, sl_te = build_seq(test_d)
print(f'  Test: {len(X_te)} seqs, median slowdown: {np.median(y_tr):.2f}')

ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te))
d = len(X_tr[0][0])
print(f'  Max seq len: {ml}, feat dim: {d}')

Xa = np.zeros((len(X_tr), ml, d), dtype=np.float32)
for i, s in enumerate(X_tr):
    Xa[i, :len(s)] = s
Xta = np.zeros((len(X_te), ml, d), dtype=np.float32)
for i, s in enumerate(X_te):
    Xta[i, :len(s)] = s

mask = np.zeros_like(Xa)
for i, s in enumerate(X_tr):
    mask[i, :len(s)] = 1.0
Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

yl = np.log(1 + np.array(y_tr, dtype=np.float32))
ym, ys = float(yl.mean()), float(yl.std()) + 1e-8

np.savez(os.path.join(OUT, 'train_data_comp_mix.npz'),
         X=(Xa - Xm) / Xs, lengths=np.array([len(s) for s in X_tr], dtype=np.int32),
         y=(yl - ym) / ys, serial_lat=np.array(sl_tr, dtype=np.float32),
         y_mean=ym, y_std=ys)
np.savez(os.path.join(OUT, 'test_data_comp_mix.npz'),
         X=(Xta - Xm) / Xs, lengths=np.array([len(s) for s in X_te], dtype=np.int32),
         y=(np.log(1 + np.array(y_te, dtype=np.float32)) - ym) / ys,
         serial_lat=np.array(sl_te, dtype=np.float32), y_mean=ym, y_std=ys)
print('Saved!')
