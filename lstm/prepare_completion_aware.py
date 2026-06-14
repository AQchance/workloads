"""
Completion-Aware Bi-LSTM training data.

Injects query COMPLETION events into the interaction sequence,
allowing the LSTM to explicitly model resource release.

Each concurrent query Q_i produces TWO events:
  - SUBMISSION at t_i_start: query begins using resources
  - COMPLETION at t_i_end: query releases resources (if t_i_end < t_j_end)

Event type flag: 0=submission, 1=completion (added as 1 extra feature dim).
"""

import sys, os, json, csv, math, numpy as np

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'


def resource_conflict(t_res, c_res):
    t = np.array(t_res); c = np.array(c_res)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_dataset(trace_file, output_prefix='train_data_completion'):
    with open(FEATURES_FILE) as f: gnf = json.load(f)

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
            if qi not in gnf or sti == 'penalty': continue
            serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
            sl = math.log(1 + serial_lat)
            qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [sl]
            t_res = list(gnf[qi]['gpu_resources'].values())

            # Build event list: submission + completion for each concurrent query
            events = []
            for oq in ov:
                if oq not in gnf: continue
                o_start = [ts for ts, _, q, _, _ in timeline if q == oq]
                o_end   = [te for _, te, q, _, _ in timeline if q == oq]
                if not o_start: continue
                os_val = o_start[0]; oe_val = o_end[0]
                # Submission event at os_val
                events.append((os_val, oq, 0))  # 0 = submission
                # Completion event at oe_val (only if before target query ends)
                if oe_val < ei:
                    events.append((oe_val, oq, 1))  # 1 = completion
            events.sort()  # sort by timestamp

            seq = []
            for et, oq, etype in events:
                osl_val = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
                ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [osl_val]
                oc = list(gnf[oq]['gpu_resources'].values())
                c = resource_conflict(t_res, oc)
                # Feature: same as before + event_type flag
                feat = qv + ovv + [si - et, 1.0 if et < si else 0.0] + c + [float(etype)]
                seq.append(feat)
            if seq:
                X.append(seq)
                y_ratio.append(rti / serial_lat)
                slats.append(serial_lat)
        return X, y_ratio, slats

    X_tr, y_tr, sl_tr = build_seq(train_d)
    X_te, y_te, sl_te = build_seq(test_d)
    print(f"  Train: {len(X_tr)} seqs, Test: {len(X_te)} seqs")
    print(f"  Ratio: {min(y_tr):.2f}-{max(y_tr):.2f}, median={np.median(y_tr):.2f}")

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='collect_concurrent/trace_2.csv')
    parser.add_argument('--output-prefix', default='train_data_completion')
    args = parser.parse_args()
    build_dataset(args.trace, args.output_prefix)
