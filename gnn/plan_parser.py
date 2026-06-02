"""
Parse TiDB EXPLAIN plans into PyG Data objects with distributed-aware features.

Each plan tree is converted to a directed graph where:
- Nodes = physical operators with 17-dim raw feature vectors (was 13)
- Edges = parent→child data flow with 5-dim edge features (was 4)

Distributed enhancements over the base parser:
  - Engine type per operator (TiDB_Server / TiKV / TiFlash)
  - Per-table data skew ratio (from TiFlash segment distribution)
  - Column-level statistics (NDV, correlation) from stats_histograms
  - Cross-engine edge flag for TiKV↔TiDB↔TiFlash data flow
  - Plan-level distributed topology features (n_tikv_ops, n_tiflash_ops, etc.)
"""

import os
import re
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

# ─── Operator class mapping (6 semantic classes) ───
OPERATOR_CLASS_MAP = {
    # SCAN
    "TableFullScan": "SCAN",
    "TableRangeScan": "SCAN",
    "IndexRangeScan": "SCAN",
    "TableRowIDScan": "SCAN",
    "IndexLookUp": "SCAN",
    "IndexReader": "SCAN",
    # JOIN
    "HashJoin": "JOIN",
    "IndexHashJoin": "JOIN",
    "IndexJoin": "JOIN",
    "MergeJoin": "JOIN",
    # AGG
    "HashAgg": "AGG",
    "StreamAgg": "AGG",
    # EXCHANGE
    "ExchangeSender": "EXCHANGE",
    "ExchangeReceiver": "EXCHANGE",
    # SORT
    "Sort": "SORT",
    "TopN": "SORT",
    # WINDOW
    "Window": "SORT",
    # FILTER
    "Projection": "FILTER",
    "Selection": "FILTER",
}

OP_CLASS_TO_ID = {"SCAN": 0, "JOIN": 1, "AGG": 2, "EXCHANGE": 3, "SORT": 4, "FILTER": 5}
N_OP_CLASSES = len(OP_CLASS_TO_ID)

# ─── Engine type (more granular than location) ───
ENGINE_TYPE_TO_ID = {"tidb_server": 0, "tikv": 1, "tiflash": 2}
N_ENGINE_TYPES = len(ENGINE_TYPE_TO_ID)

# Backward-compat: location_id mapping (used by existing code)
LOCATION_TO_ID = {"root": 0, "mpp[tiflash]": 1, "cop[tikv]": 2, "tiflash": 1}
N_LOCATIONS = len(set(LOCATION_TO_ID.values()))

JOIN_TYPE_TO_ID = {"inner": 0, "anti": 1, "semi": 2, "left": 3, "right": 4, "none": 5}
N_JOIN_TYPES = len(JOIN_TYPE_TO_ID)

EXCHANGE_TYPE_TO_ID = {"HashPartition": 0, "Broadcast": 1, "PassThrough": 2, "none": 3}
N_EXCHANGE_TYPES = len(EXCHANGE_TYPE_TO_ID)

# ─── Feature dimensions (raw indices before embedding expansion) ───
NODE_RAW_FEAT_DIM = 17   # was 13
EDGE_RAW_FEAT_DIM = 5    # was 4


# ======================================================================
# Helper parsing functions
# ======================================================================

def _normalize_op_name(raw: str) -> str:
    """Strip tree-drawing chars, trailing ID number, and Build/Probe annotations."""
    cleaned = re.sub(r'^[│├└─\s]+', '', raw)
    cleaned = re.sub(r'\(Build\)|\(Probe\)', '', cleaned).strip()
    cleaned = re.sub(r'_\d+$', '', cleaned)
    return cleaned


def _parse_op_class(op_name: str) -> Tuple[int, str]:
    """Map operator name to (class_id, class_name)."""
    cls_name = OPERATOR_CLASS_MAP.get(op_name, "FILTER")
    return OP_CLASS_TO_ID[cls_name], cls_name


def _parse_est_rows(raw: str) -> float:
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return 1.0


# ─── TPC-H table alias → full name mapping ───
TPCH_TABLE_ALIASES = {
    # Standard aliases seen in TiDB EXPLAIN
    "l": "lineitem", "lineitem": "lineitem",
    "o": "orders", "orders": "orders",
    "ps": "partsupp", "partsupp": "partsupp",
    "p": "part", "part": "part",
    "s": "supplier", "supplier": "supplier",
    "c": "customer", "customer": "customer",
    "n": "nation", "nation": "nation",
    "r": "region", "region": "region",
}


def _resolve_table_name(short_name: str) -> Optional[str]:
    """Resolve abbreviated table names to full TPC-H table names."""
    if not short_name:
        return None
    name = short_name.strip().lower()
    return TPCH_TABLE_ALIASES.get(name, name)


def _parse_task_to_engine(task_str: str) -> Tuple[int, int]:
    """Parse the 'task' column into (engine_type_id, location_id).

    engine_type_id: 0=TiDB_Server(root), 1=TiKV(cop), 2=TiFlash(mpp)
    """
    t = task_str.strip().lower()
    if "tiflash" in t:
        return ENGINE_TYPE_TO_ID["tiflash"], LOCATION_TO_ID["mpp[tiflash]"]
    if "tikv" in t:
        return ENGINE_TYPE_TO_ID["tikv"], LOCATION_TO_ID["cop[tikv]"]
    # "root" = TiDB Server
    return ENGINE_TYPE_TO_ID["tidb_server"], LOCATION_TO_ID.get(t, 0)


def _parse_table_name(access_obj: str, op_info: str = "") -> Optional[str]:
    """Extract and resolve table name.

    Tries:
      1. 'table:X' in access object (may be alias like 'l', 'o')
      2. 'schema.table.column' references in operator_info as fallback
    """
    table = None
    # Try access object first
    if access_obj:
        m = re.search(r'table:(\w+(?:\.\w+)?)', access_obj)
        if m:
            raw = m.group(1)
            if '.' in raw:
                table = raw.split('.')[-1]
            else:
                table = raw

    # Fallback: try operator_info for schema.table.column references
    if not table and op_info:
        m = re.search(r'tpch_sf40\.(\w+)\.', op_info)
        if m:
            table = m.group(1)

    return _resolve_table_name(table)


def _parse_join_type(op_info: str) -> int:
    if not op_info:
        return JOIN_TYPE_TO_ID["none"]
    m = re.search(r'(inner|anti|semi|left|right)\s+join', op_info, re.IGNORECASE)
    if m:
        return JOIN_TYPE_TO_ID.get(m.group(1).lower(), JOIN_TYPE_TO_ID["none"])
    return JOIN_TYPE_TO_ID["none"]


def _parse_exchange_type(op_info: str) -> int:
    if not op_info:
        return EXCHANGE_TYPE_TO_ID["none"]
    m = re.search(r'ExchangeType:\s*(\w+)', op_info)
    if m:
        return EXCHANGE_TYPE_TO_ID.get(m.group(1), EXCHANGE_TYPE_TO_ID["none"])
    return EXCHANGE_TYPE_TO_ID["none"]


def _parse_stream_count(op_info: str) -> int:
    if not op_info:
        return 1
    m = re.search(r'stream_count:\s*(\d+)', op_info)
    if m:
        return int(m.group(1))
    return 1


def _count_equi_conds(op_info: str) -> int:
    if not op_info:
        return 0
    return len(re.findall(r'eq\(', op_info))


def _count_group_keys(op_info: str) -> int:
    if not op_info or 'group by:' not in op_info:
        return 0
    m = re.search(r'group by:(.*?)(?:,\s*funcs:|$)', op_info)
    if m:
        group_part = m.group(1).strip()
        if group_part:
            return len(group_part.split(','))
    return 0


def _count_sort_keys(op_info: str) -> int:
    if not op_info:
        return 0
    cols = re.findall(r'(?:\w+\.\w+\.\w+|Column#\d+)', op_info)
    return len(cols)


def _has_filter(op_info: str) -> int:
    if not op_info:
        return 0
    if re.search(r'pushed down filter:(?!\s*empty)', op_info):
        return 1
    return 0


def _is_build_side(raw_id_column: str) -> int:
    return 1 if '(Build)' in raw_id_column else 0


# ======================================================================
# Plan text parser
# ======================================================================

def parse_plan_text(plan_text: str,
                    dist_cache: Optional[Dict] = None) -> Dict:
    """
    Parse a single EXPLAIN plan text block into structured data.

    Args:
        plan_text: Raw text of one plan (TAB-separated EXPLAIN FORMAT='verbose' output)
        dist_cache: Optional DistributedFeatureCollector cache with
                    keys 'table_skew' and 'column_stats'

    Returns:
        Dict with keys: nodes, edges, root_idx, n_nodes, max_depth,
                        n_tikv_ops, n_tiflash_ops, n_tidb_ops
    """
    lines = plan_text.strip().split('\n')

    # Find the plan lines (TAB-separated rows, skip header and non-plan text)
    plan_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, comments, and the header row
        if not stripped:
            continue
        if stripped.startswith('--'):
            continue
        if stripped.startswith('id\t'):
            continue
        if stripped.startswith('Execution Plan:'):
            continue
        # Plan lines contain TAB and tree-drawing characters or operator names
        if '\t' in line:
            # Verify this looks like a plan line: has tab-separated fields
            parts = line.lstrip(' │├└─').split('\t')
            if len(parts) >= 4:
                plan_lines.append(line)

    if not plan_lines:
        return {"nodes": [], "edges": [], "root_idx": 0,
                "n_nodes": 0, "max_depth": 0,
                "n_tikv_ops": 0, "n_tiflash_ops": 0, "n_tidb_ops": 0}

    # ─── Parse each plan line ───
    nodes = []

    for line in plan_lines:
        stripped_line = line.lstrip(' │├└─')
        leading_chars = len(line) - len(stripped_line)
        depth = leading_chars // 2

        # TAB-separated columns: id | estRows | [estCost] | task | access_object | operator_info
        parts = stripped_line.split('\t')
        raw_id = parts[0].strip() if len(parts) > 0 else ""

        # Handle both 5-column (no estCost) and 6-column (with estCost) formats
        if len(parts) >= 5:
            # Check if the 3rd column looks like a cost (numeric) or task (text)
            col2 = parts[1].strip() if len(parts) > 1 else ""
            col3 = parts[2].strip() if len(parts) > 2 else ""
            col4 = parts[3].strip() if len(parts) > 3 else ""
            col5 = parts[4].strip() if len(parts) > 4 else ""

            if col3 and not col3.startswith(('root', 'mpp[', 'cop[')):
                # 6-column format: id | estRows | estCost | task | access_object | operator_info
                est_rows_str = col2
                task_str = col4
                access_obj = col5
                op_info = parts[5].strip() if len(parts) > 5 else ""
            else:
                # 5-column format: id | estRows | task | access_object | operator_info
                est_rows_str = col2
                task_str = col3
                access_obj = col4
                op_info = col5
        else:
            est_rows_str = parts[1].strip() if len(parts) > 1 else "1.0"
            task_str = parts[2].strip() if len(parts) > 2 else "root"
            access_obj = parts[3].strip() if len(parts) > 3 else ""
            op_info = parts[4].strip() if len(parts) > 4 else ""

        op_name = _normalize_op_name(raw_id)
        op_class_id, op_class_name = _parse_op_class(op_name)
        engine_type_id, location_id = _parse_task_to_engine(task_str)
        table_name = _parse_table_name(access_obj, op_info) if op_class_name == "SCAN" else None

        # ─── Distributed features from cache ───
        table_skew_log = 0.0
        n_tiflash_insts = 0
        avg_col_corr = 0.0

        if table_name and dist_cache:
            skew = dist_cache.get("table_skew", {}).get(table_name, {})
            skew_ratio = skew.get("skew_ratio", 1.0)
            skew_ratio = min(skew_ratio, 100.0)  # cap
            table_skew_log = math.log(1.0 + skew_ratio)
            n_tiflash_insts = skew.get("n_instances", 0)

            # Average column correlation (proxy for sortedness → scan efficiency)
            cols = dist_cache.get("column_stats", {}).get(table_name, [])
            if cols:
                corrs = [abs(c.get("correlation", 0.0)) for c in cols]
                avg_col_corr = sum(corrs) / len(corrs)

        node = {
            "op_name": op_name,
            "op_class_id": op_class_id,
            "op_class": op_class_name,
            "est_rows": _parse_est_rows(est_rows_str),
            "location_id": location_id,
            "engine_type_id": engine_type_id,
            "stream_count": _parse_stream_count(op_info),
            "depth": depth,
            # Operator-specific
            "join_type_id": _parse_join_type(op_info),
            "exchange_type_id": _parse_exchange_type(op_info),
            "n_equi_conds": _count_equi_conds(op_info),
            "n_group_keys": _count_group_keys(op_info),
            "n_sort_keys": _count_sort_keys(op_info),
            "has_filter": _has_filter(op_info),
            "is_build_side": _is_build_side(raw_id),
            # Distributed features (per-scan-node, zero for non-scan)
            "table_name": table_name or "",
            "table_skew_log": table_skew_log,
            "n_tiflash_instances": n_tiflash_insts,
            "avg_column_correlation": avg_col_corr,
            # Raw fields preserved
            "_raw_id": raw_id,
            "_op_info": op_info,
        }
        nodes.append(node)

    # ─── Build parent-child edges from indent stack ───
    edges = []
    indent_stack = []

    for idx, node in enumerate(nodes):
        depth = node["depth"]
        while indent_stack and indent_stack[-1][0] >= depth:
            indent_stack.pop()
        if indent_stack:
            parent_idx = indent_stack[-1][1]
            edges.append((parent_idx, idx))
        indent_stack.append((depth, idx))

    # children_count
    children_count = [0] * len(nodes)
    for p, c in edges:
        children_count[p] += 1
    for idx, node in enumerate(nodes):
        node["children_count"] = children_count[idx]

    # ─── Compute subtree_est_rows (post-order) ───
    children_map = {i: [] for i in range(len(nodes))}
    for p, c in edges:
        children_map[p].append(c)

    subtree_est_rows = [0.0] * len(nodes)

    def _compute_subtree(idx: int) -> float:
        child_sum = sum(_compute_subtree(c) for c in children_map[idx])
        subtree_est_rows[idx] = nodes[idx]["est_rows"] + child_sum
        return subtree_est_rows[idx]

    has_parent = {c for _, c in edges}
    roots = [i for i in range(len(nodes)) if i not in has_parent]
    for r in roots:
        _compute_subtree(r)

    for idx, node in enumerate(nodes):
        node["subtree_est_rows"] = subtree_est_rows[idx]

    # ─── depth_ratio ───
    max_depth_val = max(n["depth"] for n in nodes) if nodes else 1
    for node in nodes:
        node["depth_ratio"] = node["depth"] / max(max_depth_val, 1)

    # ─── Plan-level engine counts ───
    n_tikv_ops = sum(1 for n in nodes if n["engine_type_id"] == ENGINE_TYPE_TO_ID["tikv"])
    n_tiflash_ops = sum(1 for n in nodes if n["engine_type_id"] == ENGINE_TYPE_TO_ID["tiflash"])
    n_tidb_ops = sum(1 for n in nodes if n["engine_type_id"] == ENGINE_TYPE_TO_ID["tidb_server"])

    return {
        "nodes": nodes,
        "edges": edges,
        "root_idx": roots[0] if roots else 0,
        "n_nodes": len(nodes),
        "max_depth": max_depth_val,
        "n_tikv_ops": n_tikv_ops,
        "n_tiflash_ops": n_tiflash_ops,
        "n_tidb_ops": n_tidb_ops,
    }


# ======================================================================
# Plan → PyG Data conversion
# ======================================================================

def plan_to_pyg_data(plan_data: Dict) -> Data:
    """Convert parsed plan data to a PyG Data object."""
    nodes = plan_data["nodes"]
    edges = plan_data["edges"]
    n = len(nodes)

    if n == 0:
        return Data(x=torch.zeros(0, NODE_RAW_FEAT_DIM),
                    edge_index=torch.zeros(2, 0, dtype=torch.long))

    # ─── Node feature matrix (17 dims) ───
    x_list = []
    for node in nodes:
        feats = []

        # 0: op_class_id
        feats.append(float(node["op_class_id"]))
        # 1: est_rows_log
        feats.append(math.log(1.0 + node["est_rows"]))
        # 2: location_id (backward compat)
        feats.append(float(node["location_id"]))
        # 3: stream_count
        feats.append(float(node["stream_count"]))
        # 4: children_count
        feats.append(float(node["children_count"]))
        # 5: depth_ratio
        feats.append(node["depth_ratio"])
        # 6: n_equi_conds
        feats.append(float(node["n_equi_conds"]))
        # 7: n_group_keys
        feats.append(float(node["n_group_keys"]))
        # 8: n_sort_keys
        feats.append(float(node["n_sort_keys"]))
        # 9: has_filter
        feats.append(float(node["has_filter"]))
        # 10: subtree_est_rows_log
        feats.append(math.log(1.0 + node["subtree_est_rows"]))
        # 11: join_type_id
        feats.append(float(node["join_type_id"]))
        # 12: exchange_type_id
        feats.append(float(node["exchange_type_id"]))
        # ─── NEW distributed features ───
        # 13: engine_type_id (0=TiDB_Server, 1=TiKV, 2=TiFlash)
        feats.append(float(node["engine_type_id"]))
        # 14: table_skew_log
        feats.append(node["table_skew_log"])
        # 15: n_tiflash_instances
        feats.append(float(node["n_tiflash_instances"]))
        # 16: avg_column_correlation
        feats.append(node["avg_column_correlation"])

        assert len(feats) == NODE_RAW_FEAT_DIM, \
            f"Expected {NODE_RAW_FEAT_DIM} features, got {len(feats)}"
        x_list.append(feats)

    x = torch.tensor(x_list, dtype=torch.float32)

    # ─── Edge index and edge features (5 dims) ───
    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges],
                                  dtype=torch.long).t().contiguous()

        edge_attr_list = []
        for p, c in edges:
            parent = nodes[p]
            child = nodes[c]
            e_feats = []

            # branch_ratio
            if parent["op_class"] == "JOIN":
                ratio = child["est_rows"] / max(parent["est_rows"], 1.0)
            else:
                ratio = 1.0
            e_feats.append(ratio)

            # loc_pair
            loc_pair = parent["location_id"] * N_LOCATIONS + child["location_id"]
            e_feats.append(float(loc_pair))

            # exchange_type from child
            e_feats.append(float(child["exchange_type_id"]))

            # is_build_side
            e_feats.append(float(child["is_build_side"]))

            # ─── NEW: cross_engine flag ───
            cross = 0 if parent["engine_type_id"] == child["engine_type_id"] else 1
            e_feats.append(float(cross))

            assert len(e_feats) == EDGE_RAW_FEAT_DIM, \
                f"Expected {EDGE_RAW_FEAT_DIM} edge features, got {len(e_feats)}"
            edge_attr_list.append(e_feats)

        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_RAW_FEAT_DIM, dtype=torch.float32)

    root_mask = torch.zeros(n, dtype=torch.bool)
    root_mask[plan_data["root_idx"]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        root_mask=root_mask,
        n_nodes=n,
    )


# ======================================================================
# File-level parsing
# ======================================================================

def parse_plan_file(filepath: str,
                    dist_cache: Optional[Dict] = None) -> List[Dict]:
    """Parse a plan file into a list of plan dicts.

    Supports two formats:
      1. "--- Line N ---" blocks (multi-plan file, legacy)
      2. "-- Query: N" header (single-plan file, current)
    """
    with open(filepath, "r") as f:
        content = f.read()

    results = []

    # ─── Format 1: multi-plan "--- Line N ---" blocks ───
    if re.search(r'^--- Line \d+ ---$', content, flags=re.MULTILINE):
        blocks = re.split(r'^--- Line (\d+) ---$', content, flags=re.MULTILINE)
        for i in range(1, len(blocks), 2):
            if i + 1 >= len(blocks):
                break
            try:
                line_no = int(blocks[i].strip())
            except ValueError:
                continue
            block_text = blocks[i + 1].strip()
            if not block_text:
                continue
            parts = block_text.split("Execution Plan:", 1)
            sql_text = parts[0].strip() if parts else ""
            plan_text = "Execution Plan:\n" + parts[1] if len(parts) > 1 else ""
            plan_data = parse_plan_text(plan_text, dist_cache=dist_cache)
            if plan_data["n_nodes"] == 0:
                continue
            pyg_data = plan_to_pyg_data(plan_data)
            results.append({
                "line_no": line_no,
                "sql": sql_text,
                "plan_data": plan_data,
                "pyg_data": pyg_data,
            })
        return results

    # ─── Format 2: single-query per file ("-- Query: N") ───
    m = re.search(r'^-- Query:\s*(\d+)$', content, flags=re.MULTILINE)
    if m:
        line_no = int(m.group(1))
        plan_text = content
        plan_data = parse_plan_text(plan_text, dist_cache=dist_cache)
        if plan_data["n_nodes"] > 0:
            pyg_data = plan_to_pyg_data(plan_data)
            results.append({
                "line_no": line_no,
                "sql": "",
                "plan_data": plan_data,
                "pyg_data": pyg_data,
            })

    return results


def parse_all_plans(plan_dir: str,
                    dist_cache: Optional[Dict] = None) -> Dict[str, List[Dict]]:
    """Parse all Q*_plan.txt files in a directory."""
    all_plans = {}
    for fname in sorted(os.listdir(plan_dir)):
        if not fname.endswith("_plan.txt") and not fname.endswith(".txt"):
            continue
        # Handle both naming conventions
        if fname.endswith("_plan.txt"):
            template = fname.replace("_plan.txt", "")
        else:
            template = fname.replace(".txt", "")
        filepath = os.path.join(plan_dir, fname)
        plans = parse_plan_file(filepath, dist_cache=dist_cache)
        if plans:
            all_plans[template] = plans
        print(f"  {fname}: {len(plans)} plans parsed")

    return all_plans


# ─── Standalone test ───
if __name__ == "__main__":
    import sys

    # Try to load distributed features
    dist_cache = None
    try:
        from distributed_features import DistributedFeatureCollector
        collector = DistributedFeatureCollector()
        dist_cache = collector.collect()
        print("Loaded distributed features from TiDB\n")
    except Exception as e:
        print(f"Note: Distributed features not available ({e}), using zero defaults\n")

    plan_dir = os.path.join(os.path.dirname(__file__), "..", "explain_plans")
    if not os.path.isdir(plan_dir):
        print(f"ERROR: plan directory not found: {plan_dir}")
        sys.exit(1)

    print(f"Parsing plans from: {plan_dir}\n")

    all_plans = parse_all_plans(plan_dir, dist_cache=dist_cache)

    total = sum(len(v) for v in all_plans.values())
    print(f"\nTotal: {total} plans across {len(all_plans)} templates")

    # Print summary with engine counts
    print("\n" + "=" * 70)
    print("Per-template summary (first plan, with engine breakdown):")
    print("=" * 70)
    for template in sorted(all_plans.keys()):
        entry = all_plans[template][0]
        pd = entry["plan_data"]
        d = entry["pyg_data"]
        print(f"  {template:5s}  line={entry['line_no']:2d}  "
              f"nodes={pd['n_nodes']:2d}  "
              f"tikv={pd['n_tikv_ops']}  tiflash={pd['n_tiflash_ops']}  "
              f"tidb={pd['n_tidb_ops']}  "
              f"x={list(d.x.shape)}  edges={d.edge_index.shape[1]}")

    # Show distributed features for the first plan with a scan node
    print("\n" + "=" * 70)
    print("Distributed feature sample (first plan with TiFlash scan):")
    print("=" * 70)
    for template in sorted(all_plans.keys()):
        entry = all_plans[template][0]
        pd = entry["plan_data"]
        scans_with_dist = [n for n in pd["nodes"]
                           if n["op_class"] == "SCAN" and n["engine_type_id"] == 2]
        if scans_with_dist:
            for n in scans_with_dist[:3]:
                print(f"  {template}  {n['op_name']}  table={n['table_name']}  "
                      f"skew_log={n['table_skew_log']:.2f}  "
                      f"n_tiflash={n['n_tiflash_instances']}  "
                      f"col_corr={n['avg_column_correlation']:.3f}")
            break
