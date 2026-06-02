"""
Parse TiDB EXPLAIN plans into PyG Data objects with node/edge features.

Each plan tree is converted to a directed graph where:
- Nodes = physical operators with 65-dim feature vectors
- Edges = parent→child data flow with 18-dim edge features
"""

import os
import re
import math
from typing import List, Dict, Tuple

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
    # FILTER
    "Projection": "FILTER",
    "Selection": "FILTER",
}

OP_CLASS_TO_ID = {"SCAN": 0, "JOIN": 1, "AGG": 2, "EXCHANGE": 3, "SORT": 4, "FILTER": 5}
N_OP_CLASSES = len(OP_CLASS_TO_ID)

LOCATION_TO_ID = {"root": 0, "mpp[tiflash]": 1, "cop[tikv]": 2, "tiflash": 1}
N_LOCATIONS = len(set(LOCATION_TO_ID.values()))

JOIN_TYPE_TO_ID = {"inner": 0, "anti": 1, "semi": 2, "left": 3, "right": 4, "none": 5}
N_JOIN_TYPES = len(JOIN_TYPE_TO_ID)

EXCHANGE_TYPE_TO_ID = {"HashPartition": 0, "Broadcast": 1, "PassThrough": 2, "none": 3}
N_EXCHANGE_TYPES = len(EXCHANGE_TYPE_TO_ID)

# ─── Feature dimensions (raw indices before embedding expansion) ───
# The model encoder expands these 13 raw features to 65+ via embedding lookups
NODE_RAW_FEAT_DIM = 13
EDGE_RAW_FEAT_DIM = 4


def _normalize_op_name(raw: str) -> str:
    """Strip tree-drawing chars, trailing ID number, and Build/Probe annotations."""
    # Remove leading tree chars and whitespace
    cleaned = re.sub(r'^[│├└─\s]+', '', raw)
    # Remove trailing _(Build) or _(Probe)
    cleaned = re.sub(r'\(Build\)|\(Probe\)', '', cleaned).strip()
    # Remove _<digits> ID suffix to get operator name
    cleaned = re.sub(r'_\d+$', '', cleaned)
    return cleaned


def _parse_op_class(op_name: str) -> Tuple[int, str]:
    """Map operator name to (class_id, class_name)."""
    cls_name = OPERATOR_CLASS_MAP.get(op_name, "FILTER")
    return OP_CLASS_TO_ID[cls_name], cls_name


def _parse_est_rows(raw: str) -> float:
    """Parse estRows, handling edge cases."""
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return 1.0


def _parse_location(raw: str) -> int:
    """Map location string to int ID."""
    loc = raw.strip()
    if "tiflash" in loc.lower():
        return LOCATION_TO_ID["mpp[tiflash]"]
    if "tikv" in loc.lower():
        return LOCATION_TO_ID["cop[tikv]"]
    return LOCATION_TO_ID.get(loc, 0)


def _parse_join_type(op_info: str) -> int:
    """Extract join type from operator_info."""
    if not op_info:
        return JOIN_TYPE_TO_ID["none"]
    m = re.search(r'(inner|anti|semi|left|right)\s+join', op_info, re.IGNORECASE)
    if m:
        return JOIN_TYPE_TO_ID.get(m.group(1).lower(), JOIN_TYPE_TO_ID["none"])
    return JOIN_TYPE_TO_ID["none"]


def _parse_exchange_type(op_info: str) -> int:
    """Extract ExchangeType from operator_info."""
    if not op_info:
        return EXCHANGE_TYPE_TO_ID["none"]
    m = re.search(r'ExchangeType:\s*(\w+)', op_info)
    if m:
        return EXCHANGE_TYPE_TO_ID.get(m.group(1), EXCHANGE_TYPE_TO_ID["none"])
    return EXCHANGE_TYPE_TO_ID["none"]


def _parse_stream_count(op_info: str) -> int:
    """Extract stream_count from operator_info."""
    if not op_info:
        return 1
    m = re.search(r'stream_count:\s*(\d+)', op_info)
    if m:
        return int(m.group(1))
    return 1


def _count_equi_conds(op_info: str) -> int:
    """Count the number of equal conditions in a join."""
    if not op_info:
        return 0
    return len(re.findall(r'eq\(', op_info))


def _count_group_keys(op_info: str) -> int:
    """Count group by columns in an aggregation."""
    if not op_info or 'group by:' not in op_info:
        return 0
    # group by:col1, col2, ..., funcs:...
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    if m:
        group_part = m.group(1).strip()
        if group_part:
            return len(group_part.split(','))
    return 0


def _count_sort_keys(op_info: str) -> int:
    """Count sort keys in a Sort or TopN operator."""
    if not op_info:
        return 0
    # For Sort: just comma-separated column references
    # For TopN: column:desc, column, ..., offset:N, count:N
    # Count only column references (containing a dot or Column#)
    cols = re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)
    return len(cols)


def _has_filter(op_info: str) -> int:
    """Check if a SCAN operator has a pushed-down filter."""
    if not op_info:
        return 0
    if re.search(r'pushed down filter:(?!\s*empty)', op_info):
        return 1
    return 0


def _is_build_side(raw_id_column: str) -> int:
    """Check if this node is the Build (inner) side of a hash join."""
    return 1 if '(Build)' in raw_id_column else 0


def _parse_memory_str(raw: str) -> float:
    """Parse memory string like '11.5 KB', '157.7 KB', '0 Bytes', 'N/A' to bytes."""
    raw = raw.strip()
    if raw.upper() == 'N/A' or raw == '':
        return 0.0
    m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB|TB)', raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        multipliers = {"BYTES": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        return val * multipliers.get(unit, 1)
    return 0.0


def parse_plan_text(plan_text: str) -> Dict:
    """
    Parse a single EXPLAIN plan text block into structured data.

    Args:
        plan_text: Raw text of one plan (from "--- Line N ---" to next "--- Line N ---")

    Returns:
        Dict with keys: nodes (List[Dict]), edges (List[Tuple[int,int]]), root_idx (int)
    """
    lines = plan_text.strip().split('\n')

    # Find the plan lines (after "Execution Plan:")
    plan_lines = []
    in_plan = False
    for line in lines:
        if 'Execution Plan:' in line:
            in_plan = True
            continue
        if in_plan and line.strip() and '|' in line:
            plan_lines.append(line)

    if not plan_lines:
        return {"nodes": [], "edges": [], "root_idx": 0}

    # ─── Parse each plan line ───
    nodes = []
    indent_stack = []  # (indent_level, node_index)

    for line in plan_lines:
        # Calculate indent depth from leading tree-drawing characters
        stripped_line = line.lstrip(' │├└─')
        leading_chars = len(line) - len(stripped_line)
        depth = leading_chars // 2

        # Split by '|'
        parts = [p.strip() for p in stripped_line.split('|')]
        if len(parts) < 5:
            continue

        raw_id = parts[0]
        est_rows_str = parts[1] if len(parts) > 1 else "1.0"
        location_str = parts[2] if len(parts) > 2 else "root"
        op_info = parts[4] if len(parts) > 4 else ""

        op_name = _normalize_op_name(raw_id)
        op_class_id, op_class_name = _parse_op_class(op_name)

        node = {
            "op_name": op_name,
            "op_class_id": op_class_id,
            "op_class": op_class_name,
            "est_rows": _parse_est_rows(est_rows_str),
            "location_id": _parse_location(location_str),
            "stream_count": _parse_stream_count(op_info),
            "depth": depth,
            # Operator-specific features
            "join_type_id": _parse_join_type(op_info),
            "exchange_type_id": _parse_exchange_type(op_info),
            "n_equi_conds": _count_equi_conds(op_info),
            "n_group_keys": _count_group_keys(op_info),
            "n_sort_keys": _count_sort_keys(op_info),
            "has_filter": _has_filter(op_info),
            "is_build_side": _is_build_side(raw_id),
            # Raw fields preserved for debugging
            "_raw_id": raw_id,
            "_op_info": op_info,
        }
        nodes.append(node)

    # ─── Build parent-child edges from indent stack ───
    edges = []
    indent_stack = []

    for idx, node in enumerate(nodes):
        depth = node["depth"]

        # Pop from stack until we find the parent
        while indent_stack and indent_stack[-1][0] >= depth:
            indent_stack.pop()

        if indent_stack:
            parent_idx = indent_stack[-1][1]
            edges.append((parent_idx, idx))

        indent_stack.append((depth, idx))

    # Set children_count
    children_count = [0] * len(nodes)
    for p, c in edges:
        children_count[p] += 1
    for idx, node in enumerate(nodes):
        node["children_count"] = children_count[idx]

    # ─── Compute subtree_est_rows (post-order) ───
    # Build adjacency for post-order traversal
    children_map = {i: [] for i in range(len(nodes))}
    for p, c in edges:
        children_map[p].append(c)

    subtree_est_rows = [0.0] * len(nodes)

    def _compute_subtree(idx: int) -> float:
        child_sum = sum(_compute_subtree(c) for c in children_map[idx])
        subtree_est_rows[idx] = nodes[idx]["est_rows"] + child_sum
        return subtree_est_rows[idx]

    # Root is the node with no parent
    has_parent = {c for _, c in edges}
    roots = [i for i in range(len(nodes)) if i not in has_parent]
    for r in roots:
        _compute_subtree(r)

    for idx, node in enumerate(nodes):
        node["subtree_est_rows"] = subtree_est_rows[idx]

    # ─── Compute depth_ratio ───
    max_depth = max(node["depth"] for node in nodes) if nodes else 1
    for node in nodes:
        node["depth_ratio"] = node["depth"] / max(max_depth, 1)

    return {
        "nodes": nodes,
        "edges": edges,
        "root_idx": roots[0] if roots else 0,
        "n_nodes": len(nodes),
        "max_depth": max_depth,
    }


def plan_to_pyg_data(plan_data: Dict) -> Data:
    """
    Convert parsed plan data to a PyG Data object with node and edge features.

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr
    """
    nodes = plan_data["nodes"]
    edges = plan_data["edges"]
    n = len(nodes)

    if n == 0:
        return Data(x=torch.zeros(0, NODE_RAW_FEAT_DIM), edge_index=torch.zeros(2, 0, dtype=torch.long))

    # ─── Build node feature matrix ───
    x_list = []
    for node in nodes:
        feats = []

        # op_class embedding index (will be embedded later in model)
        feats.append(float(node["op_class_id"]))

        # est_rows_log
        est_log = math.log(1.0 + node["est_rows"])
        feats.append(est_log)

        # location_id
        feats.append(float(node["location_id"]))

        # stream_count
        feats.append(float(node["stream_count"]))

        # children_count
        feats.append(float(node["children_count"]))

        # depth_ratio
        feats.append(node["depth_ratio"])

        # n_equi_conds
        feats.append(float(node["n_equi_conds"]))

        # n_group_keys
        feats.append(float(node["n_group_keys"]))

        # n_sort_keys
        feats.append(float(node["n_sort_keys"]))

        # has_filter
        feats.append(float(node["has_filter"]))

        # subtree_est_rows (log)
        subtree_log = math.log(1.0 + node["subtree_est_rows"])
        feats.append(subtree_log)

        # join_type_id
        feats.append(float(node["join_type_id"]))

        # exchange_type_id
        feats.append(float(node["exchange_type_id"]))

        assert len(feats) == NODE_RAW_FEAT_DIM, f"Expected {NODE_RAW_FEAT_DIM} features, got {len(feats)}"
        x_list.append(feats)

    x = torch.tensor(x_list, dtype=torch.float32)

    # ─── Build edge index and edge features ───
    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()

        edge_attr_list = []
        for p, c in edges:
            parent = nodes[p]
            child = nodes[c]

            e_feats = []

            # branch_ratio: child.estRows / parent.estRows (only meaningful for JOIN parent)
            if parent["op_class"] == "JOIN":
                ratio = child["est_rows"] / max(parent["est_rows"], 1.0)
            else:
                ratio = 1.0
            e_feats.append(ratio)

            # cross_location: encode (parent_loc, child_loc) pair
            loc_pair = parent["location_id"] * N_LOCATIONS + child["location_id"]
            e_feats.append(float(loc_pair))

            # exchange_type from child (for EXCHANGE nodes)
            e_feats.append(float(child["exchange_type_id"]))

            # is_build_side
            e_feats.append(float(child["is_build_side"]))

            assert len(e_feats) == EDGE_RAW_FEAT_DIM, f"Expected {EDGE_RAW_FEAT_DIM} edge features, got {len(e_feats)}"
            edge_attr_list.append(e_feats)

        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_RAW_FEAT_DIM, dtype=torch.float32)

    # Build root_mask (True for the root node)
    root_mask = torch.zeros(n, dtype=torch.bool)
    root_mask[plan_data["root_idx"]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        root_mask=root_mask,
        n_nodes=n,
    )


def parse_plan_file(filepath: str) -> List[Dict]:
    """
    Parse a Q*_plan.txt file, returning a list of plan dicts.

    Each dict contains:
      - line_no: original line number from .sql file
      - sql: the SQL text
      - plan_data: output of parse_plan_text()
      - pyg_data: output of plan_to_pyg_data()
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split by "--- Line N ---"
    blocks = re.split(r'^--- Line (\d+) ---$', content, flags=re.MULTILINE)

    results = []
    # blocks[0] is text before first "--- Line N ---" (usually empty)
    # Then pairs: [line_no_1, block_1, line_no_2, block_2, ...]
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

        # Separate SQL text from plan
        parts = block_text.split("Execution Plan:", 1)
        sql_text = parts[0].strip() if parts else ""
        plan_text = "Execution Plan:\n" + parts[1] if len(parts) > 1 else ""

        plan_data = parse_plan_text(plan_text)
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


def parse_all_plans(plan_dir: str) -> Dict[str, List[Dict]]:
    """
    Parse all Q*_plan.txt files in a directory.

    Returns:
        Dict mapping template name (e.g., "Q1") to list of plan dicts.
    """
    all_plans = {}
    for fname in sorted(os.listdir(plan_dir)):
        if not fname.endswith("_plan.txt"):
            continue
        template = fname.replace("_plan.txt", "")
        filepath = os.path.join(plan_dir, fname)
        plans = parse_plan_file(filepath)
        if plans:
            all_plans[template] = plans
        print(f"  {fname}: {len(plans)} plans parsed")

    return all_plans


# ─── Standalone test ───
if __name__ == "__main__":
    import sys

    plan_dir = os.path.join(os.path.dirname(__file__), "..", "plans")
    if not os.path.isdir(plan_dir):
        print(f"ERROR: plan directory not found: {plan_dir}")
        sys.exit(1)

    print(f"Parsing plans from: {plan_dir}\n")

    all_plans = parse_all_plans(plan_dir)

    total = sum(len(v) for v in all_plans.values())
    print(f"\nTotal: {total} plans across {len(all_plans)} templates")

    # Print summary for first plan of each template
    print("\n" + "=" * 70)
    print("Per-template summary (first plan):")
    print("=" * 70)
    for template in sorted(all_plans.keys()):
        entry = all_plans[template][0]
        pd = entry["plan_data"]
        d = entry["pyg_data"]
        print(f"  {template:5s}  line={entry['line_no']:2d}  "
              f"nodes={pd['n_nodes']:2d}  max_depth={pd['max_depth']:2d}  "
              f"x={list(d.x.shape)}  edges={d.edge_index.shape[1]}")

    # Verify feature ranges
    print("\n" + "=" * 70)
    print("Feature range check (first Q1 plan):")
    print("=" * 70)
    q1_entry = all_plans.get("Q1", [])[0] if all_plans.get("Q1") else None
    if q1_entry:
        x = q1_entry["pyg_data"].x
        for i in range(NODE_RAW_FEAT_DIM):
            col = x[:, i]
            print(f"  feat[{i:2d}]: min={col.min().item():12.4f}  max={col.max().item():12.4f}  mean={col.mean().item():12.4f}")
