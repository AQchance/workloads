"""
Build pure ICONQ baseline training data.

ICONQ interaction feature (Section 3.1):
  x_i = [iconq_feat(Q_i) || iconq_feat(Q_j) || |t_i - t_j| || is_before]

NO GNN plan_emb, NO GNN resource predictions, NO resource conflict signals.
This is the true ICONQ ablation baseline.
"""

import sys, os, json, csv, math, numpy as np

OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'


def build_dataset(trace_file, iconq_features_file, output_prefix='train_data_iconq_pure'):
    with open(iconq_features_file) as f:
        iconq_features = json.load(f)

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
            if qi not in iconq_features or sti == 'penalty': continue
            serial_lat = max(iconq_features[qi]['serial_labels'].get('latency_s', 1), 0.5)
            iconq_vec = iconq_features[qi]['iconq_feat']  # 59 dims, incl. runtime feature

            seq = []; oi = []
            for oq in ov:
                if oq not in iconq_features: continue
                for os, _, _, _, _ in [(ts, te, q, _, _) for ts, te, q, _, _ in timeline if q == oq]:
                    oi.append((os, oq)); break
            oi.sort()
            for os, oq in oi:
                ovv = iconq_features[oq]['iconq_feat']  # 59 dims
                # Pure ICONQ: only concatenate query features + timestamp
                feat = iconq_vec + ovv + [si - os, 1.0 if os < si else 0.0]
                seq.append(feat)
            if seq:
                X.append(seq)
                y_ratio.append(rti / serial_lat)
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

    np.savez(os.path.join(OUT_DIR, f'{output_prefix}.npz'),
             X=(Xa - Xm) / Xs, lengths=tr_len,
             y=(y_log - ym) / ys, serial_lat=sl_tr_arr,
             y_mean=ym, y_std=ys)
    test_prefix = output_prefix.replace('train', 'test')
    np.savez(os.path.join(OUT_DIR, test_prefix + '.npz'),
             X=(Xta - Xm) / Xs, lengths=te_len,
             y=(np.log(1 + np.array(y_te, dtype=np.float32)) - ym) / ys,
             serial_lat=sl_te_arr,
             y_mean=ym, y_std=ys)

    print(f"  Saved: {output_prefix}.npz")
    return d


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='collect_concurrent/trace_2.csv')
    parser.add_argument('--features', default='lstm/iconq_features.json')
    parser.add_argument('--output-prefix', default='train_data_iconq_pure')
    args = parser.parse_args()
    build_dataset(args.trace, args.features, args.output_prefix)
