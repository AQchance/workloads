"""
ICONQ-style baseline: absolute runtime prediction (not ratio).

ICONQ query feature: 1(predicted serial latency) + 2×n_p operators + n_t tables
Prediction target: log(1 + concurrent_runtime_seconds)
"""

import os, re, json, csv, math, numpy as np, sys

OP_TYPES = [
    'TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
    'IndexLookUp', 'IndexReader',
    'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
    'HashAgg', 'StreamAgg',
    'Sort', 'TopN', 'Window',
    'ExchangeSender', 'ExchangeReceiver',
    'Projection', 'Selection',
]

TPCH_TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier',
               'customer', 'nation', 'region']


def extract_query_feature(plan_text: str, predicted_lat: float) -> list:
    """ICONQ-style feature: predicted latency + operator stats + table cardinalities."""
    op_counts = {op: 0 for op in OP_TYPES}
    op_est_rows = {op: 0.0 for op in OP_TYPES}
    table_card = {t: 0.0 for t in TPCH_TABLES}

    for line in plan_text.split('\n'):
        if '\t' not in line or line.startswith('--'):
            continue
        stripped = line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) < 5: continue
        raw_id = parts[0].strip()
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        try: est_rows = float(parts[1].strip())
        except ValueError: est_rows = 1.0
        if op_name in op_counts:
            op_counts[op_name] += 1
            op_est_rows[op_name] += est_rows
        op_info = parts[4].strip() if len(parts) > 4 else ''
        for tbl in TPCH_TABLES:
            if tbl in op_info.lower():
                table_card[tbl] = max(table_card[tbl], est_rows)

    feat = [math.log(1 + predicted_lat)]  # Stage-equivalent: GNN-predicted latency
    for op in OP_TYPES:
        feat.append(float(op_counts[op]))
        feat.append(math.log(1 + op_est_rows[op]))
    for tbl in TPCH_TABLES:
        feat.append(math.log(1 + table_card[tbl]))
    return feat


def extract_all(trace_file, plan_dir, gnn_features_file, out_file):
    with open(gnn_features_file) as f: gnf = json.load(f)

    trace_qids = set()
    with open(trace_file) as f:
        for row in csv.DictReader(f):
            trace_qids.add(row['qid'])

    feats = {}
    for qid in sorted(trace_qids):
        pf = os.path.join(plan_dir, f'{qid}.txt')
        if not os.path.exists(pf): continue
        with open(pf) as f: plan_text = f.read()
        # ICONQ uses PREDICTED latency, not actual (Stage in their case, GNN in ours)
        gres = gnf.get(qid, {}).get('gpu_resources', {})
        predicted_lat = max(abs(float(gres.get('lat', 0.5))), 0.5)
        feat = extract_query_feature(plan_text, predicted_lat)
        serial_actual = gnf.get(qid, {}).get('serial_labels', {}).get('latency_s', 1)
        feats[qid] = {'iconq_feat': feat, 'serial_lat': serial_actual}

    with open(out_file, 'w') as f: json.dump(feats, f)
    print(f"Extracted {len(feats)} queries, dim={len(list(feats.values())[0]['iconq_feat'])} -> {out_file}")


if __name__ == '__main__':
    trace_f = sys.argv[1] if len(sys.argv) > 1 else 'collect_concurrent/trace_2.csv'
    plan_d = sys.argv[2] if len(sys.argv) > 2 else 'explain_plans'
    gnn_f  = sys.argv[3] if len(sys.argv) > 3 else 'lstm/gnn_features.json'
    out_f  = sys.argv[4] if len(sys.argv) > 4 else 'lstm/iconq_features_v2.json'
    extract_all(trace_f, plan_d, gnn_f, out_f)
