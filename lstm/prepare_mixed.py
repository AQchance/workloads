"""
Build mixed K=2+K=4 training data for Bi-LSTM.
Combines both traces, time-splits 70/30, saves combined npz.
"""

import sys, os, json, csv, math, numpy as np

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'
TRACES = [
    '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv',
    '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_4.csv',
]

def resource_conflict(t_res, c_res):
    t = np.array(t_res); c = np.array(c_res)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))

with open(FEATURES_FILE) as f: gnf = json.load(f)

# Load and merge both traces
all_timeline = []
for tf in TRACES:
    with open(tf) as f:
        trace = list(csv.DictReader(f))
    for r in trace:
        qid = r['qid']; rt = float(r['runtime']); st = r['status']
        actual = 60.0 if st == 'penalty' else rt
        all_timeline.append((float(r['start']), float(r['start']) + actual, qid, actual, st))

# Sort by start time, compute overlaps
all_timeline.sort()
ci = []
for i, (si, ei, qi, rti, sti) in enumerate(all_timeline):
    ov = []
    for j, (sj, ej, qj, _, _) in enumerate(all_timeline):
        if i != j and sj < ei and ej > si: ov.append(qj)
    ci.append((qi, si, ei, rti, sti, ov))

mt = max(e for _, _, e, _, _, _ in ci); sp = mt * 0.7
tr_d = [c for c in ci if c[1] < sp]; te_d = [c for c in ci if c[1] >= sp]

def build_seq(data_list):
    X, y_ratio, slats = [], [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        sl = math.log(1 + serial_lat)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [sl]
        t_res = list(gnf[qi]['gpu_resources'].values())
        seq = []; oi = []
        for oq in ov:
            if oq not in gnf: continue
            for os, _, _, _, _ in [(ts, te, q, _, _) for ts, te, q, _, _ in all_timeline if q == oq]:
                oi.append((os, oq)); break
        oi.sort()
        for os, oq in oi:
            osl = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [osl]
            oc = list(gnf[oq]['gpu_resources'].values()); c = resource_conflict(t_res, oc)
            seq.append(qv + ovv + [si - os, 1.0 if os < si else 0.0] + c)
        if seq:
            X.append(seq); y_ratio.append(rti / serial_lat); slats.append(serial_lat)
    return X, y_ratio, slats

X_tr, y_tr, sl_tr = build_seq(tr_d); X_te, y_te, sl_te = build_seq(te_d)
print(f"Train: {len(X_tr)} seqs, Test: {len(X_te)} seqs")
print(f"Ratio range: {min(y_tr):.2f}-{max(y_tr):.2f}, median={np.median(y_tr):.2f}")

ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te)); d = len(X_tr[0][0])
Xa = np.zeros((len(X_tr), ml, d), dtype=np.float32)
for i, s in enumerate(X_tr): Xa[i, :len(s)] = s
Xta = np.zeros((len(X_te), ml, d), dtype=np.float32)
for i, s in enumerate(X_te): Xta[i, :len(s)] = s
mask = np.zeros_like(Xa)
for i, s in enumerate(X_tr): mask[i, :len(s)] = 1.0
Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
diff = ((Xa - Xm) * mask) ** 2
Xs = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
y_log = np.log(1 + np.array(y_tr, dtype=np.float32)); ym, ys = y_log.mean(), y_log.std() + 1e-8

tr_len = np.array([len(s) for s in X_tr], dtype=np.int32)
te_len = np.array([len(s) for s in X_te], dtype=np.int32)

np.savez(os.path.join(OUT_DIR, 'train_data_mixed.npz'),
         X=(Xa - Xm) / Xs, lengths=tr_len,
         y=(y_log - ym) / ys, y_mean=ym, y_std=ys,
         serial_lat=np.array(sl_tr, dtype=np.float32))
np.savez(os.path.join(OUT_DIR, 'test_data_mixed.npz'),
         X=(Xta - Xm) / Xs, lengths=te_len,
         y=(np.log(1+np.array(y_te,dtype=np.float32))-ym)/ys,
         y_mean=ym, y_std=ys,
         serial_lat=np.array(sl_te, dtype=np.float32))

print(f"Saved: train_data_mixed.npz ({os.path.getsize(os.path.join(OUT_DIR,'train_data_mixed.npz'))/1024:.0f}KB)")
print(f"Saved: test_data_mixed.npz ({os.path.getsize(os.path.join(OUT_DIR,'test_data_mixed.npz'))/1024:.0f}KB)")
