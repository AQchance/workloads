"""
Build training data with ratio labels + resource conflict vectors.

Labels: slowdown_ratio = concurrent_runtime / serial_latency
Features: interaction vectors with per-pair resource conflict (5 dims).
"""

import sys, os, json, csv, math, numpy as np

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'


def resource_conflict(t_res, c_res):
    t = np.array(t_res); c = np.array(c_res)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_dataset(trace_file, features_file):
    with open(features_file) as f: gnn_features = json.load(f)

    trace = []
    with open(trace_file) as f:
        for row in csv.DictReader(f): trace.append(row)

    timeline = []
    for r in trace:
        qid = r['qid']; rt = float(r['runtime']); st = r['status']
        actual = 60.0 if st == 'penalty' else rt
        s = float(r['start'])
        timeline.append((s, s + actual, qid, actual, st))

    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = []
        for j, (sj, ej, qj, _, _) in enumerate(timeline):
            if i != j and sj < ei and ej > si: ov.append(qj)
        ci.append((qi, si, ei, rti, sti, ov))

    mt = max(e for _, _, e, _, _, _ in ci); sp = mt * 0.7
    train_d = [c for c in ci if c[1] < sp]
    test_d  = [c for c in ci if c[1] >= sp]

    def build_seq(data_list):
        X, y_ratio, slats = [], [], []
        for qi, si, ei, rti, sti, ov in data_list:
            if qi not in gnn_features or sti == 'penalty': continue
            serial_lat = max(gnn_features[qi]['serial_labels'].get('latency_s', 1), 0.5)
            sl = math.log(1 + serial_lat)
            qv = gnn_features[qi]['plan_emb'] + list(gnn_features[qi]['gpu_resources'].values()) + [sl]
            t_res = list(gnn_features[qi]['gpu_resources'].values())
            seq = []; oi = []
            for oq in ov:
                if oq not in gnn_features: continue
                for os, _, _, _, _ in [(ts, te, q, _, _) for ts, te, q, _, _ in timeline if q == oq]:
                    oi.append((os, oq)); break
            oi.sort()
            for os, oq in oi:
                osl = math.log(1 + gnn_features[oq]['serial_labels'].get('latency_s', 10))
                ovv = gnn_features[oq]['plan_emb'] + list(gnn_features[oq]['gpu_resources'].values()) + [osl]
                oc = list(gnn_features[oq]['gpu_resources'].values())
                c = resource_conflict(t_res, oc)
                feat = qv + ovv + [si - os, 1.0 if os < si else 0.0] + c
                seq.append(feat)
            if seq:
                X.append(seq)
                y_ratio.append(rti / serial_lat)   # ratio label
                slats.append(serial_lat)
        return X, y_ratio, slats

    X_tr, y_tr, sl_tr = build_seq(train_d)
    X_te, y_te, sl_te = build_seq(test_d)
    print(f"  Train: {len(X_tr)} seqs, Test: {len(X_te)} seqs")
    print(f"  Ratio range: {min(y_tr):.2f} - {max(y_tr):.2f}, median={np.median(y_tr):.2f}")

    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te))
    d = len(X_tr[0][0])
    print(f"  Max seq len: {ml}, feat dim: {d}")

    Xa = np.zeros((len(X_tr), ml, d), dtype=np.float32)
    for i, s in enumerate(X_tr): Xa[i, :len(s)] = s
    Xta = np.zeros((len(X_te), ml, d), dtype=np.float32)
    for i, s in enumerate(X_te): Xta[i, :len(s)] = s

    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_tr): mask[i, :len(s)] = 1.0
    Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
    diff = ((Xa - Xm) * mask) ** 2
    Xs = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

    y_log = np.log(1 + np.array(y_tr, dtype=np.float32))
    ym, ys = y_log.mean(), y_log.std() + 1e-8

    tr_len = np.array([len(s) for s in X_tr], dtype=np.int32)
    te_len = np.array([len(s) for s in X_te], dtype=np.int32)
    sl_tr_arr = np.array(sl_tr, dtype=np.float32)
    sl_te_arr = np.array(sl_te, dtype=np.float32)

    # Save
    np.savez(os.path.join(OUT_DIR, 'train_data.npz'),
             X=(Xa - Xm) / Xs, lengths=tr_len,
             y=(y_log - ym) / ys, serial_lat=sl_tr_arr,
             y_mean=ym, y_std=ys)
    np.savez(os.path.join(OUT_DIR, 'test_data.npz'),
             X=(Xta - Xm) / Xs, lengths=te_len,
             y=(np.log(1 + np.array(y_te, dtype=np.float32)) - ym) / ys,
             serial_lat=sl_te_arr,
             y_mean=ym, y_std=ys)

    print(f"  Saved: train_data.npz ({os.path.getsize(os.path.join(OUT_DIR, 'train_data.npz'))/1024:.0f} KB)")
    print(f"  Saved: test_data.npz ({os.path.getsize(os.path.join(OUT_DIR, 'test_data.npz'))/1024:.0f} KB)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv',
                        help='Concurrent trace CSV file (trace_2.csv or trace_4.csv)')
    args = parser.parse_args()
    build_dataset(args.trace, FEATURES_FILE)
