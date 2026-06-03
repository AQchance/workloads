"""
Train PlanGNN with NDV-based memory features + distributed topology features.

Node features (20 dims):
  13 original + 3 NDV (join_mem_log, agg_mem_log, sort_mem_log)
  + 4 distributed (engine_type_id, table_skew_log, n_tiflash_instances, column_corr)

Edge features (5 dims): 4 original + cross_engine flag

Global skip (16 dims): 9 scalar sums + n_nodes + 3 dist sums + 3 engine counts
"""

import os, sys, re, math, json, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN

# ─── Constants ───
NODE_RAW_DIM = 22  # 13 base + 3 NDV + 4 distributed + 2 exchange_bytes
EDGE_RAW_DIM = 5   # 4 base + cross_engine

N_LOCATIONS = 3

# ─── Operator mappings ───
OPERATOR_CLASS_MAP = {
    "TableFullScan": 0, "TableRangeScan": 0, "IndexRangeScan": 0,
    "TableRowIDScan": 0, "IndexLookUp": 0, "IndexReader": 0,
    "HashJoin": 1, "IndexHashJoin": 1, "IndexJoin": 1, "MergeJoin": 1,
    "HashAgg": 2, "StreamAgg": 2,
    "ExchangeSender": 3, "ExchangeReceiver": 3,
    "Sort": 4, "TopN": 4, "Window": 4,
    "Projection": 5, "Selection": 5,
}
LOCATION_MAP = {"root": 0, "mpp[tiflash]": 1, "cop[tikv]": 2, "tiflash": 1}
JOIN_TYPE_MAP = {"inner": 0, "anti": 1, "semi": 2, "left": 3, "right": 4, "none": 5}
EXCHANGE_TYPE_MAP = {"HashPartition": 0, "Broadcast": 1, "PassThrough": 2, "none": 3}
ENGINE_TYPE_MAP = {"tidb_server": 0, "tikv": 1, "tiflash": 2}


def _engine_type_from_task(task_str: str) -> int:
    """0=TiDB_Server, 1=TiKV, 2=TiFlash."""
    t = task_str.strip().lower()
    if "tiflash" in t: return 2
    if "tikv" in t: return 1
    return 0


def _resolve_table(access_obj: str, op_info: str, aliases: dict) -> Optional[str]:
    """Extract and resolve table name from EXPLAIN output."""
    t = None
    if access_obj:
        m = re.search(r'table:(\w+(?:\.\w+)?)', access_obj)
        if m:
            raw = m.group(1)
            t = raw.split('.')[-1] if '.' in raw else raw
    if not t and op_info:
        m = re.search(r'tpch_sf40\.(\w+)\.', op_info)
        if m: t = m.group(1)
    if t:
        t = t.strip().lower()
        return aliases.get(t, t)
    return None


def load_ndv_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_dist_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_join_columns(op_info: str) -> list:
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info)]


def parse_group_columns(op_info: str) -> list:
    if 'group by:' not in op_info:
        return []
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    if not m:
        return []
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', m.group(1))]


def compute_ndv_features(lines: list, ndv_cache: dict) -> list:
    """Compute per-node (join_mem_log, agg_mem_log, sort_mem_log)."""
    results = []
    for line in lines:
        stripped = line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) < 5:
            results.append((0.0, 0.0, 0.0))
            continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip() if len(parts) > 1 else "1.0"
        op_info = parts[4].strip() if len(parts) > 4 else ""

        try: est_rows = max(float(est_rows_str), 1.0)
        except ValueError: est_rows = 1.0

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)

        join_mem, agg_mem, sort_mem = 0.0, 0.0, 0.0

        if op_name in ('HashJoin', 'IndexHashJoin', 'IndexJoin', 'MergeJoin'):
            cols = parse_join_columns(op_info)
            if cols:
                ndvs, widths = [], []
                for col in cols:
                    info = ndv_cache.get(col, {"ndv": est_rows, "avg_width": 8})
                    ndvs.append(info.get("ndv", est_rows))
                    widths.append(info.get("avg_width", 8))
                n_equi = len(re.findall(r'eq\(', op_info))
                min_ndv = min(ndvs) if ndvs else est_rows
                avg_width = sum(widths) / len(widths) if widths else 8
                hash_entries = min(est_rows, min_ndv)
                join_mem = hash_entries * avg_width * (1.0 + 0.1 * n_equi)

        elif op_name in ('HashAgg', 'StreamAgg'):
            group_cols = parse_group_columns(op_info)
            if group_cols:
                ndvs = []
                for col in group_cols:
                    info = ndv_cache.get(col, {"ndv": est_rows, "avg_width": 8})
                    ndvs.append(info.get("ndv", est_rows))
                n_group = len(group_cols)
                agg_ndv = min(ndvs) if ndvs else est_rows
                agg_mem = min(est_rows, agg_ndv) * 8 * n_group

        elif op_name in ('Sort', 'TopN'):
            n_sort = max(len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)), 1)
            sort_mem = est_rows * n_sort * 8

        results.append((
            math.log(1.0 + join_mem),
            math.log(1.0 + agg_mem),
            math.log(1.0 + sort_mem),
        ))

    return results


def _compute_subtree_columns(lines: list, start_idx: int) -> set:
    """Collect all tpch_sf40.table.column references in a node's subtree."""
    stripped_start = lines[start_idx].lstrip(' │├└─')
    start_depth = (len(lines[start_idx]) - len(stripped_start)) // 2
    cols = set()
    for j in range(start_idx + 1, len(lines)):
        stripped = lines[j].lstrip(' │├└─')
        depth = (len(lines[j]) - len(stripped)) // 2
        if depth <= start_depth:
            break
        parts = stripped.split('\t')
        if len(parts) < 5:
            continue
        op_info = parts[4].strip() if len(parts) > 4 else ''
        for tbl, col in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info):
            cols.add(f'{tbl}.{col}')
    return cols


def _exchange_row_width(cols: set, ndv_cache: dict) -> float:
    """Compute row width from column set using NDV cache avg_width."""
    width = 0
    for col_key in cols:
        info = ndv_cache.get(col_key, {'avg_width': 8})
        width += info.get('avg_width', 8)
    return width if width > 0 else 8


def parse_plan(plan_text: str, ndv_cache: dict,
               dist_cache: Optional[dict] = None) -> Optional[Data]:
    """Parse EXPLAIN plan into PyG Data with NDV + distributed features."""
    lines = [l for l in plan_text.strip().split('\n') if '\t' in l and not l.startswith('--')]
    if not lines:
        return None

    ndv_feats = compute_ndv_features(lines, ndv_cache)
    assert len(ndv_feats) == len(lines), f"NDV feat count mismatch: {len(ndv_feats)} vs {len(lines)}"

    aliases = dist_cache.get("table_aliases", {}) if dist_cache else {}
    table_skew = dist_cache.get("table_skew", {}) if dist_cache else {}
    col_stats = dist_cache.get("column_stats", {}) if dist_cache else {}

    nodes, indent_stack = [], []
    for idx, line in enumerate(lines):
        stripped = line.lstrip(' │├└─')
        depth = (len(line) - len(stripped)) // 2
        parts = stripped.split('\t')
        if len(parts) < 3:
            continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip()
        location_str = parts[2].strip()
        access_obj = parts[3].strip() if len(parts) > 3 else ""
        op_info = parts[4].strip() if len(parts) > 4 else ""

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        op_class = OPERATOR_CLASS_MAP.get(op_name, 5)

        try: est_rows = float(est_rows_str)
        except ValueError: est_rows = 1.0

        loc_id = LOCATION_MAP.get(location_str, 0)
        eng_id = _engine_type_from_task(location_str)
        stream = int(re.search(r'stream_count:\s*(\d+)', op_info).group(1)) if 'stream_count' in op_info else 1
        jt = JOIN_TYPE_MAP.get(
            (re.search(r'(inner|anti|semi|left|right)\s+join', op_info, re.I) or [None, 'none'])[1].lower(), 5)
        et = EXCHANGE_TYPE_MAP.get(
            (re.search(r'ExchangeType:\s*(\w+)', op_info) or [None, 'none'])[1], 3)
        ne = len(re.findall(r'eq\(', op_info))
        ng = (len(re.search(r'group by:(.*?)(?:, funcs:|$)', op_info).group(1).split(','))
              if 'group by:' in op_info else 0)
        ns = len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)) if op_class == 4 else 0
        hf = 1 if re.search(r'pushed down filter:(?!\s*empty)', op_info) else 0
        ib = 1 if '(Build)' in raw_id else 0

        jml, aml, sml = ndv_feats[idx]

        # ─── Distributed features ───
        table_name = None
        if op_class == 0:  # SCAN
            table_name = _resolve_table(access_obj, op_info, aliases)

        t_skew_log = 0.0
        n_tif = 0
        col_corr = 0.0
        if table_name and table_skew:
            skew_info = table_skew.get(table_name, {})
            t_skew_log = math.log(1.0 + min(skew_info.get("skew_ratio", 1.0), 100.0))
            n_tif = skew_info.get("n_instances", 0)
            c_info = col_stats.get(table_name, {})
            col_corr = c_info.get("avg_correlation", 0.0)

        # ─── Exchange bytes features (computed after all nodes parsed) ───
        exch_row_width_log = 0.0
        exch_est_bytes_log = 0.0

        nodes.append({
            "depth": depth, "op_class": op_class, "est_rows": est_rows,
            "location_id": loc_id, "engine_type_id": eng_id,
            "stream_count": stream,
            "join_type": jt, "exchange_type": et,
            "n_equi": ne, "n_group": ng, "n_sort": ns,
            "has_filter": hf, "is_build": ib,
            "join_mem_log": jml, "agg_mem_log": aml, "sort_mem_log": sml,
            "table_skew_log": t_skew_log,
            "n_tiflash_instances": n_tif,
            "column_corr": col_corr,
            # Exchange bytes (filled below for Exchange nodes)
            "exch_row_width_log": exch_row_width_log,
            "exch_est_bytes_log": exch_est_bytes_log,
            "_line_idx": idx,
        })

    # ─── Post-process: compute Exchange byte features ───
    for n in nodes:
        if n["op_class"] == 3:  # EXCHANGE
            cols = _compute_subtree_columns(lines, n["_line_idx"])
            row_width = _exchange_row_width(cols, ndv_cache)
            n["exch_row_width_log"] = math.log(1.0 + row_width)
            n["exch_est_bytes_log"] = math.log(1.0 + n["est_rows"] * row_width)

    # Edges
    edges, indent_stack = [], []
    for idx, node in enumerate(nodes):
        depth = node["depth"]
        while indent_stack and indent_stack[-1][0] >= depth:
            indent_stack.pop()
        if indent_stack:
            edges.append((indent_stack[-1][1], idx))
        indent_stack.append((depth, idx))

    child_count = [0] * len(nodes)
    for p, c in edges: child_count[p] += 1
    for i, n in enumerate(nodes): n["children_count"] = child_count[i]

    children_map = defaultdict(list)
    for p, c in edges: children_map[p].append(c)

    def compute_subtree(i):
        s = nodes[i]["est_rows"]
        for c in children_map[i]: s += compute_subtree(c)
        nodes[i]["subtree_est"] = s
        return s

    has_parent_set = {c for _, c in edges}
    for i in range(len(nodes)):
        if i not in has_parent_set:
            compute_subtree(i)

    max_depth = max(n["depth"] for n in nodes) if nodes else 1
    for n in nodes: n["depth_ratio"] = n["depth"] / max(max_depth, 1)

    # ─── Feature matrix [N, 22] ───
    x_list = []
    for n in nodes:
        x_list.append([
            float(n["op_class"]),                     # 0
            math.log(1.0 + n["est_rows"]),             # 1
            float(n["location_id"]),                   # 2
            float(n["stream_count"]),                  # 3
            float(n["children_count"]),                # 4
            n["depth_ratio"],                          # 5
            float(n["n_equi"]),                        # 6
            float(n["n_group"]),                       # 7
            float(n["n_sort"]),                        # 8
            float(n["has_filter"]),                    # 9
            math.log(1.0 + n["subtree_est"]),          # 10
            float(n["join_type"]),                     # 11
            float(n["exchange_type"]),                 # 12
            n["join_mem_log"],                         # 13 (NDV)
            n["agg_mem_log"],                          # 14 (NDV)
            n["sort_mem_log"],                         # 15 (NDV)
            float(n["engine_type_id"]),                # 16 (distributed)
            n["table_skew_log"],                       # 17 (distributed)
            float(n["n_tiflash_instances"]),           # 18 (distributed)
            n["column_corr"],                          # 19 (distributed)
            n["exch_row_width_log"],                   # 20 (exchange bytes)
            n["exch_est_bytes_log"],                   # 21 (exchange bytes)
        ])
    x = torch.tensor(x_list, dtype=torch.float32)

    # ─── Edge features [E, 5] ───
    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        e_attr_list = []
        for p, c in edges:
            parent, child = nodes[p], nodes[c]
            ratio = child["est_rows"] / max(parent["est_rows"], 1.0) if parent["op_class"] == 1 else 1.0
            loc_pair = parent["location_id"] * N_LOCATIONS + child["location_id"]
            cross = 0 if parent["engine_type_id"] == child["engine_type_id"] else 1
            e_attr_list.append([
                ratio,                          # 0: branch_ratio
                float(loc_pair),                # 1: loc_pair
                float(child["exchange_type"]),  # 2: exchange_type
                float(child["is_build"]),       # 3: is_build
                float(cross),                   # 4: cross_engine (new)
            ])
        edge_attr = torch.tensor(e_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_RAW_DIM, dtype=torch.float32)

    root_mask = torch.zeros(len(nodes), dtype=torch.bool)
    for i in range(len(nodes)):
        if i not in has_parent_set:
            root_mask[i] = True
            break

    # ─── Plan-level engine counts ───
    n_tidb = sum(1 for n in nodes if n["engine_type_id"] == 0)
    n_tikv = sum(1 for n in nodes if n["engine_type_id"] == 1)
    n_tiflash = sum(1 for n in nodes if n["engine_type_id"] == 2)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                root_mask=root_mask, n_nodes=len(nodes),
                n_tidb=n_tidb, n_tikv=n_tikv, n_tiflash=n_tiflash)


def parse_memory_bytes(raw: str) -> Optional[float]:
    raw = raw.strip()
    if raw.upper() == 'N/A' or raw == '':
        return None
    m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB|TB)', raw, re.I)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        return val * {"BYTES": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824, "TB": 1099511627776}[unit]
    return None


def parse_time_ms(raw: str) -> float:
    if not raw: return 0.0
    m = re.search(r'time:(\d+)m([\d.]+)s', raw)
    if m: return (int(m.group(1)) * 60 + float(m.group(2))) * 1000
    m = re.search(r'time:([\d.]+)(s|ms)', raw)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val * 1000 if unit == 's' else val
    return 0.0


def parse_analyze(text: str, ndv_cache: dict = None) -> Optional[Dict]:
    """Extract query-level resource labels. Network label is bytes (not rows)."""
    lines = text.strip().split('\n')
    rows = []
    in_table = False
    for line in lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8: rows.append(parts)
    if not rows: return None

    # Build parsed rows with depth for subtree column lookup
    parsed = []
    for row in rows:
        raw_id = row[0].strip()
        stripped = raw_id.lstrip(' │├└─')
        depth = (len(raw_id) - len(stripped)) // 2
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        exec_info = row[5].strip() if len(row) > 5 else ""
        op_info = row[6].strip() if len(row) > 6 else ""
        memory_str = row[7].strip() if len(row) > 7 else "N/A"
        act_rows_str = row[2].strip() if len(row) > 2 else "0"
        try: act_rows = float(act_rows_str)
        except ValueError: act_rows = 0.0
        parsed.append({
            "depth": depth, "op": op_name, "exec_info": exec_info,
            "info": op_info, "memory": memory_str, "act_rows": act_rows,
        })

    total_mem, total_disk, total_net_bytes = 0.0, 0.0, 0.0
    for i, p in enumerate(parsed):
        op_name = p["op"]

        mem_val = parse_memory_bytes(p["memory"])
        if mem_val: total_mem += mem_val

        if op_name in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
            m = re.search(r'data_scanned_rows:(\d+)', p["exec_info"])
            if m: total_disk += float(m.group(1))

        if op_name in ("ExchangeSender", "ExchangeReceiver"):
            # Compute row width from subtree columns
            cols = set()
            for j in range(i + 1, len(parsed)):
                if parsed[j]["depth"] <= p["depth"]:
                    break
                for tbl, col in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', parsed[j]["info"]):
                    cols.add(f'{tbl}.{col}')
            row_width = 8  # default
            if ndv_cache and cols:
                row_width = sum(
                    ndv_cache.get(c, {}).get('avg_width', 8) for c in cols)
                if row_width == 0: row_width = 8
            total_net_bytes += p["act_rows"] * row_width

    cpu_time = parse_time_ms(parsed[0]["exec_info"] if parsed else "")
    return {"latency_ms": max(cpu_time, 0), "memory_bytes": total_mem,
            "disk_io_rows": total_disk, "network_bytes": total_net_bytes}


def load_dataset(plan_dir: str, analyze_dir: str, ndv_cache: dict,
                 dist_cache: Optional[dict] = None):
    """Load all aligned plan-label pairs with NDV + distributed features."""
    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    graphs, labels, meta = [], [], []
    for fname in common:
        with open(os.path.join(plan_dir, fname)) as f: plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f: analyze_text = f.read()

        g = parse_plan(plan_text, ndv_cache, dist_cache)
        lab = parse_analyze(analyze_text, ndv_cache)
        if g is None or lab is None or g.x.shape[0] == 0:
            continue
        graphs.append(g)
        labels.append(lab)
        meta.append(fname.replace('.txt', ''))

    return graphs, labels, meta


def normalize_labels(labels: List[Dict]) -> Tuple[List[Dict], Dict]:
    keys = ["latency_ms", "memory_bytes", "disk_io_rows", "network_bytes"]
    log_labels = [{k: math.log(1.0 + max(l[k], 0)) for k in keys} for l in labels]
    stats = {}
    for k in keys:
        vals = [l[k] for l in log_labels]
        stats[k] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-8)}
    return [{k: (l[k] - stats[k]["mean"]) / stats[k]["std"] for k in keys} for l in log_labels], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_plans')
    parser.add_argument('--analyze-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_analyze_results')
    parser.add_argument('--ndv-cache', default='/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
    parser.add_argument('--dist-cache', default='/home/anqian/Desktop/my_lab/workloads/dist_cache.json')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-dist', action='store_true',
                        help='Disable distributed features (baseline run)')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Loading caches...")
    ndv_cache = load_ndv_cache(args.ndv_cache)
    print(f"  NDV: {len(ndv_cache)} columns")

    dist_cache = None
    if not args.no_dist:
        try:
            dist_cache = load_dist_cache(args.dist_cache)
            print(f"  Dist: {len(dist_cache.get('table_skew',{}))} tables, "
                  f"{len(dist_cache.get('column_stats',{}))} with col stats")
        except FileNotFoundError:
            print("  Dist cache not found, running WITHOUT distributed features")
            args.no_dist = True

    print(f"Feature mode: {'BASE (16-dim, no distributed)' if args.no_dist else 'FULL (20-dim, with distributed)'}")

    print("Loading dataset...")
    graphs, labels, meta = load_dataset(args.plan_dir, args.analyze_dir, ndv_cache, dist_cache)
    print(f"  {len(graphs)} aligned plan-label pairs")

    norm_labels, stats = normalize_labels(labels)
    key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_bytes': 'net', 'latency_ms': 'cpu'}
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in key_map.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.7)]
    val_idx = indices[int(n * 0.7):int(n * 0.85)]
    test_idx = indices[int(n * 0.85):]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64)

    # ─── Model setup ───
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            preds = model(data)
            loss = sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                        for k in key_map.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                preds = model(data)
                val_loss += sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                                for k in key_map.values()).item()
                n_b += 1
        val_loss /= max(n_b, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f'E{epoch:3d} train={train_loss/n_batches:.4f} val={val_loss:.4f} best={best_val:.4f}')

    os.makedirs('checkpoints', exist_ok=True)
    ckpt_name = 'best_ndv.pt' if args.no_dist else 'best_dist.pt'
    torch.save(best_state, f'checkpoints/{ckpt_name}')
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64)
    all_p, all_t = {k: [] for k in key_map.values()}, {k: [] for k in key_map.values()}
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            preds = model(data)
            for k in key_map.values():
                all_p[k].append(preds[k].squeeze(-1).cpu().numpy())
                all_t[k].append(getattr(data, f'y_{k}').squeeze(-1).cpu().numpy())

    rmap = {'mem': 'memory_bytes', 'disk': 'disk_io_rows', 'net': 'network_bytes', 'cpu': 'latency_ms'}
    baseline = {'mem': (1.56, 11.41, 86.07, 0.78),
                'disk': (1.18, 1.93, 2.44, 0.92),
                'net': (0, 0, 0, 0),    # bytes-based, no prior baseline
                'cpu': (1.71, 4.02, 6.01, 0.59)}

    mode_label = "BASE (no dist)" if args.no_dist else "FULL (with dist)"
    print(f"\n{'Dim':>8s}  {'P50':>8s} {'Δ':>6s}  {'P90':>8s} {'Δ':>6s}  {'P95':>8s} {'Δ':>6s}  {'R²':>8s} {'Δ':>6s}  [{mode_label}]")
    print("-" * 90)
    for k, label in [('mem', 'Memory'), ('disk', 'DiskIO'), ('net', 'Network'), ('cpu', 'Latency')]:
        p = np.concatenate(all_p[k]).flatten()
        t = np.concatenate(all_t[k]).flatten()
        std, mn = stats[rmap[k]]['std'], stats[rmap[k]]['mean']
        p_raw = np.maximum(np.exp(p * std + mn) - 1, 0)
        t_raw = np.exp(t * std + mn) - 1
        qe = np.maximum(p_raw / np.maximum(t_raw, 1), np.maximum(t_raw, 1) / np.maximum(p_raw, 1))
        qs = np.sort(qe)
        nq = len(qs)
        # Safety: clip extreme Q-errors for R² stability
        p_clipped = np.clip(p, -10, 10)
        t_clipped = np.clip(t, -10, 10)
        r2 = 1 - np.sum((t_clipped - p_clipped) ** 2) / max(np.sum((t_clipped - np.mean(t_clipped)) ** 2), 1e-8)
        print(f'{label:>8s}  {qs[nq//2]:>8.2f} {qs[nq//2]-baseline[k][0]:>+6.2f}  '
              f'{qs[int(nq*0.9)]:>8.2f} {qs[int(nq*0.9)]-baseline[k][1]:>+6.2f}  '
              f'{qs[int(nq*0.95)]:>8.2f} {qs[int(nq*0.95)]-baseline[k][2]:>+6.2f}  '
              f'{r2:>8.4f} {r2-baseline[k][3]:>+6.4f}')

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint: checkpoints/{ckpt_name}")


if __name__ == '__main__':
    main()
