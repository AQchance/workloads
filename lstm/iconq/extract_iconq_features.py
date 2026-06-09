"""
Extract ICONQ-style flat features (operator counts + cardinality sums)
for all queries in the concurrent trace.

ICONQ uses ~40 dims per query:
  - For each operator type: count + sum(estRows) = ~30 dims
  - Table access flags: 8 dims (TPC-H tables)
  - Plan summary: 2 dims (total estRows, n_nodes)
  = ~40 dims total
"""

import sys, os, json, re, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gnn'))

TRACE_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv'
PLAN_DIR = '/home/anqian/Desktop/my_lab/workloads/explain_plans'
CGROUP_DIR = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
OUT_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/iconq/iconq_features.json'

# Operator types ICONQ tracks (matching their paper/code)
OPERATOR_TYPES = [
    'TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'IndexLookUp',
    'HashJoin', 'IndexHashJoin', 'IndexJoin', 'MergeJoin',
    'HashAgg', 'StreamAgg',
    'ExchangeSender', 'ExchangeReceiver',
    'Sort', 'TopN', 'Window',
    'Projection', 'Selection',
    'TableReader', 'IndexReader',
]
OP_IDX = {op: i for i, op in enumerate(OPERATOR_TYPES)}
N_OPS = len(OPERATOR_TYPES)

# TPC-H tables
TPCH_TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'customer', 'supplier', 'nation', 'region']
TABLE_IDX = {t: i for i, t in enumerate(TPCH_TABLES)}
N_TABLES = len(TPCH_TABLES)

# Table alias mapping
ALIASES = {'l': 'lineitem', 'o': 'orders', 'ps': 'partsupp', 'p': 'part',
           'c': 'customer', 's': 'supplier', 'n': 'nation', 'r': 'region'}


def extract_iconq_features(plan_text):
    """Parse EXPLAIN plan and extract ICONQ-style flat feature vector."""
    op_counts = [0.0] * N_OPS
    op_card_sums = [0.0] * N_OPS
    table_flags = [0.0] * N_TABLES
    total_estrows = 0.0
    n_nodes = 0

    in_plan = False
    for line in plan_text.split('\n'):
        if not line.strip() or line.startswith('--'):
            continue
        if line.startswith('id\t'):
            in_plan = True
            continue
        if not in_plan or '\t' not in line:
            continue

        parts = line.split('\t')
        if len(parts) < 3:
            continue
        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip()
        access_obj = parts[3].strip() if len(parts) > 3 else ''
        op_info = parts[4].strip() if len(parts) > 4 else ''

        # Parse operator name
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)

        try:
            est_rows = float(est_rows_str)
        except ValueError:
            est_rows = 1.0

        n_nodes += 1
        total_estrows += est_rows

        # Count operators
        if op_name in OP_IDX:
            idx = OP_IDX[op_name]
            op_counts[idx] += 1.0
            op_card_sums[idx] += math.log(1.0 + est_rows)

        # Table access
        if op_name in ('TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan'):
            m = re.search(r'table:(\w+)', access_obj)
            if m:
                tbl = ALIASES.get(m.group(1), m.group(1))
                if tbl in TABLE_IDX:
                    table_flags[TABLE_IDX[tbl]] = 1.0

    # Build feature vector
    features = []
    for i in range(N_OPS):
        features.append(op_counts[i])
        features.append(op_card_sums[i])
    features.extend(table_flags)
    features.append(math.log(1.0 + total_estrows))
    features.append(float(n_nodes))

    return features


# Load trace
trace_qids = set()
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f):
        trace_qids.add(row['qid'])
print(f"Unique queries in trace: {len(trace_qids)}")

# Load serial labels from cgroup
from train_cgroup import load_cgroup_labels
cgroup_labels = load_cgroup_labels(CGROUP_DIR)

# Extract features
features = {}
n_skipped = 0

for qid in sorted(trace_qids):
    pf = os.path.join(PLAN_DIR, f'{qid}.txt')
    if not os.path.exists(pf):
        n_skipped += 1
        continue
    with open(pf) as f:
        plan_text = f.read()

    vec = extract_iconq_features(plan_text)
    serial_lat = 10.0
    if qid in cgroup_labels:
        serial_lat = cgroup_labels[qid].get('latency_s', 10.0)

    features[qid] = {
        'features': vec,
        'serial_latency_s': serial_lat,
    }

print(f"Extracted features for {len(features)} queries ({len(vec)} dims), skipped {n_skipped}")

with open(OUT_FILE, 'w') as f:
    json.dump(features, f)
print(f"Saved to {OUT_FILE} ({os.path.getsize(OUT_FILE)/1024:.0f} KB)")
