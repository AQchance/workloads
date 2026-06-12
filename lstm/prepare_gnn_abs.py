"""
Build GNN training data predicting absolute concurrent runtime (seconds).
For fair ICONQ comparison: same target, same architecture, different encoding.
"""

import json, csv, math, numpy as np, os

OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'


def resource_conflict(t_res, c_res):
    t = np.array(t_res); c = np.array(c_res)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_dataset(trace_file, gnn_features_file, output_prefix='train_data_gnn_abs'):
    with open(gnn_features_file) as f: gnf = json.load(f)

    trace = []
    with open(trace_file) as f:
        for row in csv.DictReader(f): trace.append(row)

    timeline = []
    for r in trace:
        qid = r['qid']; rt = float(r['runtime']); st = r['status']
        actual = 60.0 if st == 'penalty' else rt
        timeline.append((float(r['start']), float(r['start'])+actual, qid, actual, st))

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
        X, y_abs = [], []
        for qi, si, ei, rti, sti, ov in data_list:
            if qi not in gnf or sti == 'penalty': continue
            pred_lat = max(abs(float(gnf[qi]['gpu_resources'].get('lat', 1))), 0.5)
            plan_emb = gnf[qi]['plan_emb']
            gres = list(gnf[qi]['gpu_resources'].values())
            qv = plan_emb + gres + [math.log(1+pred_lat)]
            t_res = gres
            seq = []; oi = []
            for oq in ov:
                if oq not in gnf: continue
                for os, _, _, _, _ in [(ts, te, q, _, _) for ts, te, q, _, _ in timeline if q == oq]:
                    oi.append((os, oq)); break
            oi.sort()
            for os, oq in oi:
                opred_lat = max(abs(float(gnf[oq]['gpu_resources'].get('lat', 1))), 0.5)
                ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [math.log(1+opred_lat)]
                c = resource_conflict(t_res, list(gnf[oq]['gpu_resources'].values()))
                feat = qv + ovv + [si-os, 1.0 if os < si else 0.0] + c
                seq.append(feat)
            if seq:
                X.append(seq)
                y_abs.append(rti)
        return X, y_abs

    X_tr, y_tr = build_seq(train_d)
    X_te, y_te = build_seq(test_d)
    print(f"  Train: {len(X_tr)} seqs, Test: {len(X_te)} seqs")
    print(f"  Runtime: {min(y_tr):.1f}s-{max(y_tr):.1f}s, median={np.median(y_tr):.1f}s")

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

    np.savez(os.path.join(OUT_DIR, f'{output_prefix}.npz'),
             X=(Xa - Xm) / Xs, lengths=tr_len,
             y=(y_log - ym) / ys, y_mean=ym, y_std=ys)
    test_prefix = output_prefix.replace('train', 'test')
    np.savez(os.path.join(OUT_DIR, test_prefix + '.npz'),
             X=(Xta - Xm) / Xs, lengths=te_len,
             y=(np.log(1 + np.array(y_te, dtype=np.float32)) - ym) / ys,
             y_mean=ym, y_std=ys)
    print(f"  Saved: {output_prefix}.npz")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='collect_concurrent/trace_2.csv')
    parser.add_argument('--features', default='lstm/gnn_features.json')
    parser.add_argument('--output-prefix', default='train_data_gnn_abs')
    args = parser.parse_args()
    build_dataset(args.trace, args.features, args.output_prefix)
