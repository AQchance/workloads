"""
Extract ICONQ-style flat vector features from EXPLAIN plans.

ICONQ query feature (Section 3.1):
  - Plan features: 2 × n_p (count + sum estRows for each operator type)
  - Table features: n_t (cardinality per top table)
  - Runtime feature: GNN-predicted serial latency (our Stage equivalent)

This is the baseline ablation: replace GNN plan_emb with ICONQ's
hand-crafted feature vector for the Bi-LSTM.
"""

import os, re, json, csv, math, numpy as np, sys

OP_TYPES = [
    # Scan operators
    'TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
    'IndexLookUp', 'IndexReader',
    # Join operators
    'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
    # Aggregate operators
    'HashAgg', 'StreamAgg',
    # Sort operators
    'Sort', 'TopN', 'Window',
    # Exchange operators
    'ExchangeSender', 'ExchangeReceiver',
    # Filter operators
    'Projection', 'Selection',
]
N_P = len(OP_TYPES)  # 18

N_T = 20  # top tables

TPCH_TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier',
               'customer', 'nation', 'region']


def extract_query_feature(plan_text: str, serial_lat: float) -> list:
    """Extract ICONQ-style feature from EXPLAIN plan text."""
    op_counts = {op: 0 for op in OP_TYPES}
    op_est_rows = {op: 0.0 for op in OP_TYPES}
    table_card = {t: 0.0 for t in TPCH_TABLES}

    for line in plan_text.split('\n'):
        if '\t' not in line or line.startswith('--'):
            continue
        stripped = line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) < 5:
            continue
        raw_id = parts[0].strip()
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        try:
            est_rows = float(parts[1].strip())
        except ValueError:
            est_rows = 1.0

        if op_name in op_counts:
            op_counts[op_name] += 1
            op_est_rows[op_name] += est_rows

        # Table access info
        op_info = parts[4].strip() if len(parts) > 4 else ''
        for tbl in TPCH_TABLES:
            if tbl in op_info.lower():
                table_card[tbl] = max(table_card[tbl], est_rows)

    # Build feature vector
    feat = [math.log(1 + serial_lat)]  # runtime feature (GNN prediction)

    for op in OP_TYPES:
        feat.append(float(op_counts[op]))           # count
        feat.append(math.log(1 + op_est_rows[op])) # sum estRows

    for tbl in TPCH_TABLES:
        feat.append(math.log(1 + table_card[tbl]))

    # Pad to N_T tables
    while len(feat) < 1 + 2 * N_P + N_T:
        feat.append(0.0)

    return feat


def extract_all(trace_file, plan_dir, gnn_features_file, out_file):
    """Extract ICONQ features for all queries in trace."""
    with open(gnn_features_file) as f:
        gnn_features = json.load(f)

    trace_qids = set()
    with open(trace_file) as f:
        for row in csv.DictReader(f):
            trace_qids.add(row['qid'])

    feats = {}
    for qid in sorted(trace_qids):
        pf = os.path.join(plan_dir, f'{qid}.txt')
        if not os.path.exists(pf):
            continue
        with open(pf) as f:
            plan_text = f.read()

        serial_lat = 1.0
        if qid in gnn_features:
            serial_lat = max(gnn_features[qid]['serial_labels'].get('latency_s', 1), 0.5)

        feat = extract_query_feature(plan_text, serial_lat)
        feats[qid] = {'iconq_feat': feat, 'serial_labels': gnn_features.get(qid, {}).get('serial_labels', {})}

    with open(out_file, 'w') as f:
        json.dump(feats, f)

    print(f"Extracted features for {len(feats)} queries, saved to {out_file}")
    print(f"Feature dim: {len(list(feats.values())[0]['iconq_feat'])}")


if __name__ == '__main__':
    trace_file = sys.argv[1] if len(sys.argv) > 1 else 'collect_concurrent/trace_2.csv'
    plan_dir = sys.argv[2] if len(sys.argv) > 2 else 'explain_plans'
    gnn_file = sys.argv[3] if len(sys.argv) > 3 else 'lstm/gnn_features.json'
    out_file = sys.argv[4] if len(sys.argv) > 4 else 'lstm/iconq_features.json'
    extract_all(trace_file, plan_dir, gnn_file, out_file)
