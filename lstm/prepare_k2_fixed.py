"""Prepare all 3 datasets for K=2 fixed trace from same split."""
import os, json, csv, math, numpy as np, sys

ROOT = '/home/anqian/Desktop/my_lab/workloads'
TRACE = os.path.join(ROOT, 'collect_concurrent', 'trace_2_fixed.csv')
FEATURES = os.path.join(ROOT, 'lstm', 'gnn_features_k2_fixed.json')
OUT = os.path.join(ROOT, 'lstm')
TAG = sys.argv[1] if len(sys.argv) > 1 else '_k2_fixed'

with open(FEATURES) as f: gnf = json.load(f)
iconq_feats = {}
for fn in ['lstm/iconq_features_v2.json']:
    with open(os.path.join(ROOT, fn)) as f: iconq_feats.update(json.load(f))

timeline = []
with open(TRACE) as f:
    for row in csv.DictReader(f):
        rt = float(row['runtime']); st = row['status']
        actual = 60.0 if st == 'penalty' else rt
        timeline.append((float(row['start']), float(row['start'])+actual, row['qid'], actual, st))

qid_info = {}
for s, e, q, _, _ in timeline:
    if q not in qid_info: qid_info[q] = (s, e)

ci = []
for i, (si, ei, qi, rti, sti) in enumerate(timeline):
    ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si]
    ci.append((qi, si, ei, rti, sti, ov))

mt = max(e for _, _, e, _, _, _ in ci); sp = mt * 0.7
train_d = [c for c in ci if c[1] < sp]; test_d = [c for c in ci if c[1] >= sp]
print(f'Split: {len(train_d)} train, {len(test_d)} test')

def rc(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t)+np.abs(c)+1e-8, 1e-8))

# Build GNN original
def build_gnn(dl):
    X, yr, sl = [], [], []
    for qi, si, ei, rti, sti, ov in dl:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [math.log(1+serial_lat)]
        tr = list(gnf[qi]['gpu_resources'].values())
        seq = []
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in gnf and oq in qid_info]
        oi.sort()
        for os_val, oq in oi:
            oslv = math.log(1+gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [oslv]
            c = rc(tr, list(gnf[oq]['gpu_resources'].values()))
            seq.append(qv + ovv + [si-os_val, 1.0 if os_val < si else 0.0] + c)
        if seq: X.append(seq); yr.append(rti/serial_lat); sl.append(serial_lat)
    return X, yr, sl

# Build GNN completion
def build_comp(dl):
    X, yr, sl = [], [], []
    for qi, si, ei, rti, sti, ov in dl:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [math.log(1+serial_lat)]
        tr = list(gnf[qi]['gpu_resources'].values())
        events = []
        for oq in ov:
            if oq not in gnf or oq not in qid_info: continue
            osv, oev = qid_info[oq]; events.append((osv, oq, 0))
            if oev < ei: events.append((oev, oq, 1))
        events.sort()
        seq = []
        for et, oq, etype in events:
            oslv = math.log(1+gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [oslv]
            c = rc(tr, list(gnf[oq]['gpu_resources'].values()))
            seq.append(qv + ovv + [si-et, 1.0 if et < si else 0.0] + c + [float(etype)])
        if seq: X.append(seq); yr.append(rti/serial_lat); sl.append(serial_lat)
    return X, yr, sl

# Build ICONQ
def build_iconq(dl):
    X, yr = [], []
    for qi, si, ei, rti, sti, ov in dl:
        if qi not in iconq_feats or sti == 'penalty': continue
        qv = iconq_feats[qi]['iconq_feat']
        seq = []
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in iconq_feats and oq in qid_info]
        oi.sort()
        for os_val, oq in oi:
            seq.append(qv + iconq_feats[oq]['iconq_feat'] + [si-os_val, 1.0 if os_val < si else 0.0])
        if seq: X.append(seq); yr.append(rti)
    return X, yr

def save_gnn(X_seq, yr, sl, ml, d, prefix):
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
    Xm = (Xa*mask).sum(axis=(0,1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa-Xm)**2*mask).sum(axis=(0,1))/max(mask.sum(),1)) + 1e-8
    yl = np.log(1+np.array(yr, dtype=np.float32))
    ym, ys_val = float(yl.mean()), float(yl.std())+1e-8
    np.savez(os.path.join(OUT, f'{prefix}.npz'), X=(Xa-Xm)/Xs,
             lengths=np.array([len(s) for s in X_seq], dtype=np.int32),
             y=(yl-ym)/ys_val, serial_lat=np.array(sl, dtype=np.float32),
             y_mean=ym, y_std=ys_val)
    print(f'  Saved {prefix}.npz ({len(X_seq)} seqs, dim={d}, max_len={ml})')

def save_iconq(X_seq, y_abs, ml, d, prefix):
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
    Xm = (Xa*mask).sum(axis=(0,1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa-Xm)**2*mask).sum(axis=(0,1))/max(mask.sum(),1)) + 1e-8
    yl = np.log(1+np.array(y_abs, dtype=np.float32))
    ym, ys_val = float(yl.mean()), float(yl.std())+1e-8
    np.savez(os.path.join(OUT, f'{prefix}.npz'), X=(Xa-Xm)/Xs,
             lengths=np.array([len(s) for s in X_seq], dtype=np.int32),
             y=(yl-ym)/ys_val, y_mean=ym, y_std=ys_val)
    print(f'  Saved {prefix}.npz ({len(X_seq)} seqs, dim={d}, max_len={ml})')

# Build all
print('Building GNN original...')
X_tr, y_tr, sl_tr = build_gnn(train_d); X_te, y_te, sl_te = build_gnn(test_d)
print(f'  Train: {len(X_tr)}, Test: {len(X_te)}, median ratio: {np.median(y_tr):.2f}')
ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te)); d = len(X_tr[0][0])
save_gnn(X_tr, y_tr, sl_tr, ml, d, f'train_gnn{TAG}'); save_gnn(X_te, y_te, sl_te, ml, d, f'test_gnn{TAG}')

print('Building GNN completion...')
X_tr2, y_tr2, sl_tr2 = build_comp(train_d); X_te2, y_te2, sl_te2 = build_comp(test_d)
ml2 = max(max(len(s) for s in X_tr2), max(len(s) for s in X_te2)); d2 = len(X_tr2[0][0])
save_gnn(X_tr2, y_tr2, sl_tr2, ml2, d2, f'train_comp{TAG}'); save_gnn(X_te2, y_te2, sl_te2, ml2, d2, f'test_comp{TAG}')

print('Building ICONQ...')
X_tr3, y_tr3 = build_iconq(train_d); X_te3, y_te3 = build_iconq(test_d)
ml3 = max(max(len(s) for s in X_tr3), max(len(s) for s in X_te3)); d3 = len(X_tr3[0][0])
save_iconq(X_tr3, y_tr3, ml3, d3, f'train_iconq{TAG}'); save_iconq(X_te3, y_te3, ml3, d3, f'test_iconq{TAG}')

print(f'\nDone! All saved with tag={TAG}')
