"""
Train PlanGNN with per-operator memory auxiliary supervision.

Key addition: node-level memory labels from EXPLAIN ANALYZE memory column.
Only operators with non-N/A memory are used for auxiliary loss.
"""

import os, sys, re, math, argparse, json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN

# ─── Operator mappings (same as train_sqlstorm.py) ───
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


def parse_memory_bytes(raw: str) -> Optional[float]:
    """Parse memory string to bytes. Returns None if N/A."""
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
    """Extract time in ms from execution info."""
    if not raw:
        return 0.0
    m = re.search(r'time:(\d+)m([\d.]+)s', raw)
    if m:
        return (int(m.group(1)) * 60 + float(m.group(2))) * 1000
    m = re.search(r'time:([\d.]+)(s|ms)', raw)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val * 1000 if unit == 's' else val
    return 0.0


def load_dataset_with_node_mem(plan_dir: str, analyze_dir: str):
    """
    Load plans + labels, including per-node memory labels.
    Returns: (graphs, query_labels, meta_list)
    Each graph has:
      - Usual node features x
      - y_mem, y_disk, y_net, y_cpu (query-level normalized labels)
      - node_mem_label (per-node memory in log(1+bytes), or NaN for N/A nodes)
      - node_mem_mask (bool, True where label is valid)
    """
    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    query_labels_raw = []
    graphs = []
    meta = []

    for fname in common:
        qnum = fname.replace('.txt', '')
        with open(os.path.join(plan_dir, fname)) as f:
            plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f:
            analyze_text = f.read()

        # ─── Parse EXPLAIN plan → graph ───
        g, node_mem_raw, node_mem_mask_list = _parse_plan_with_mem(plan_text, analyze_text)
        if g is None or g.x.shape[0] == 0:
            continue

        # ─── Extract query-level labels ───
        labels_raw = _parse_query_labels(analyze_text)
        if labels_raw is None:
            continue

        # Store per-node memory (log-space for labels that exist)
        node_mem_log = np.zeros(g.x.shape[0], dtype=np.float32)
        node_mem_valid = np.zeros(g.x.shape[0], dtype=bool)
        for idx, (mem_val, has_mem) in enumerate(zip(node_mem_raw, node_mem_mask_list)):
            if has_mem and mem_val is not None and not np.isnan(mem_val) and mem_val >= 0:
                node_mem_log[idx] = math.log(1.0 + mem_val)
                node_mem_valid[idx] = True

        g.node_mem_label = torch.tensor(node_mem_log, dtype=torch.float32)
        g.node_mem_mask = torch.tensor(node_mem_valid, dtype=torch.bool)

        graphs.append(g)
        query_labels_raw.append(labels_raw)
        meta.append(qnum)

    return graphs, query_labels_raw, meta


def _parse_plan_with_mem(plan_text: str, analyze_text: str):
    """Parse EXPLAIN plan into a graph, and extract per-node memory from ANALYZE."""
    # Parse plan for graph structure
    lines = plan_text.strip().split('\n')
    plan_lines = []
    for line in lines:
        if line.startswith('--') or not line.strip():
            continue
        if '\t' in line:
            plan_lines.append(line)

    if not plan_lines:
        return None, [], []

    # Parse ANALYZE for per-node memory
    analyze_lines = analyze_text.strip().split('\n')
    analyze_mem = {}  # op_id → memory_bytes (or None)
    in_table = False
    for line in analyze_lines:
        if line.startswith('--') or not line.strip():
            continue
        if line.startswith('id\t'):
            in_table = True
            continue
        if in_table and '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 8:
                raw_id = parts[0].strip()
                memory = parts[7].strip() if len(parts) > 7 else 'N/A'
                mem_val = parse_memory_bytes(memory)
                analyze_mem[raw_id] = mem_val  # None if N/A

    # Build nodes
    nodes = []
    indent_stack = []
    node_raw_ids = []  # Track raw_id for memory lookup

    for line in plan_lines:
        stripped = line.lstrip(' │├└─')
        depth = (len(line) - len(stripped)) // 2
        parts = stripped.split('\t')
        if len(parts) < 3:
            continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip() if len(parts) > 1 else "1.0"
        location_str = parts[2].strip() if len(parts) > 2 else "root"
        op_info = parts[4].strip() if len(parts) > 4 else ""

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        op_class = OPERATOR_CLASS_MAP.get(op_name, 5)

        try:
            est_rows = float(est_rows_str)
        except ValueError:
            est_rows = 1.0

        loc_id = LOCATION_MAP.get(location_str, 0)
        stream_count = int(re.search(r'stream_count:\s*(\d+)', op_info).group(1)) if 'stream_count' in op_info else 1
        join_type = JOIN_TYPE_MAP.get(
            (re.search(r'(inner|anti|semi|left|right)\s+join', op_info, re.I) or [None, 'none'])[1].lower(), 5)
        exchange_type = EXCHANGE_TYPE_MAP.get(
            (re.search(r'ExchangeType:\s*(\w+)', op_info) or [None, 'none'])[1], 3)
        n_equi = len(re.findall(r'eq\(', op_info))
        n_group = (len(re.search(r'group by:(.*?)(?:, funcs:|$)', op_info).group(1).split(','))
                   if 'group by:' in op_info else 0)
        n_sort = len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)) if op_class == 4 else 0
        has_filter = 1 if re.search(r'pushed down filter:(?!\s*empty)', op_info) else 0
        is_build = 1 if '(Build)' in raw_id else 0

        nodes.append({
            "depth": depth, "op_class": op_class, "est_rows": est_rows,
            "location_id": loc_id, "stream_count": stream_count,
            "join_type": join_type, "exchange_type": exchange_type,
            "n_equi": n_equi, "n_group": n_group, "n_sort": n_sort,
            "has_filter": has_filter, "is_build": is_build,
        })
        node_raw_ids.append(raw_id)

    # Build edges
    edges = []
    indent_stack = []
    for idx, node in enumerate(nodes):
        depth = node["depth"]
        while indent_stack and indent_stack[-1][0] >= depth:
            indent_stack.pop()
        if indent_stack:
            edges.append((indent_stack[-1][1], idx))
        indent_stack.append((depth, idx))

    # Children count
    child_count = [0] * len(nodes)
    for p, c in edges:
        child_count[p] += 1
    for i, n in enumerate(nodes):
        n["children_count"] = child_count[i]

    # Subtree estRows
    children_map = defaultdict(list)
    for p, c in edges:
        children_map[p].append(c)

    def compute_subtree(i):
        s = nodes[i]["est_rows"]
        for c in children_map[i]:
            s += compute_subtree(c)
        nodes[i]["subtree_est"] = s
        return s

    has_parent_set = {c for _, c in edges}
    for i in range(len(nodes)):
        if i not in has_parent_set:
            compute_subtree(i)

    max_depth = max(n["depth"] for n in nodes) if nodes else 1
    for n in nodes:
        n["depth_ratio"] = n["depth"] / max(max_depth, 1)

    # Build feature matrix [N, 13]
    x_list = []
    for n in nodes:
        feats = [
            float(n["op_class"]), math.log(1.0 + n["est_rows"]),
            float(n["location_id"]), float(n["stream_count"]),
            float(n["children_count"]), n["depth_ratio"],
            float(n["n_equi"]), float(n["n_group"]), float(n["n_sort"]),
            float(n["has_filter"]), math.log(1.0 + n["subtree_est"]),
            float(n["join_type"]), float(n["exchange_type"]),
        ]
        x_list.append(feats)

    x = torch.tensor(x_list, dtype=torch.float32)

    # Edge features
    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        N_LOC = 3
        e_attr_list = []
        for p, c in edges:
            parent, child = nodes[p], nodes[c]
            ratio = child["est_rows"] / max(parent["est_rows"], 1.0) if parent["op_class"] == 1 else 1.0
            loc_pair = parent["location_id"] * N_LOC + child["location_id"]
            e_attr_list.append([ratio, float(loc_pair), float(child["exchange_type"]), float(child["is_build"])])
        edge_attr = torch.tensor(e_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4, dtype=torch.float32)

    root_mask = torch.zeros(len(nodes), dtype=torch.bool)
    for i in range(len(nodes)):
        if i not in has_parent_set:
            root_mask[i] = True
            break

    # Per-node memory from ANALYZE
    # Match by index: plan lines correspond 1:1 to node_raw_ids
    node_mem_raw = []
    node_mem_mask = []
    for rid in node_raw_ids:
        # Try exact match first, then try matching without annotations
        mem = analyze_mem.get(rid)
        if mem is None:
            # Try matching by removing (Build)/(Probe) suffixes from the ANALYZE key
            for akey, aval in analyze_mem.items():
                clean_akey = re.sub(r'\(Build\)|\(Probe\)', '', akey).strip()
                clean_rid = re.sub(r'\(Build\)|\(Probe\)', '', rid).strip()
                if clean_akey == clean_rid:
                    mem = aval
                    break
        node_mem_raw.append(mem)
        node_mem_mask.append(mem is not None)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, root_mask=root_mask, n_nodes=len(nodes))
    return g, node_mem_raw, node_mem_mask


def _parse_query_labels(text: str) -> Optional[Dict]:
    """Extract query-level resource labels from EXPLAIN ANALYZE."""
    lines = text.strip().split('\n')
    rows = []
    in_table = False
    for line in lines:
        if line.startswith('id\t'):
            in_table = True
            continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8:
                rows.append(parts)

    if not rows:
        return None

    total_memory = 0.0
    total_disk = 0.0
    total_network = 0.0

    for row in rows:
        raw_id = row[0].strip()
        exec_info = row[5].strip() if len(row) > 5 else ""
        memory_str = row[7].strip() if len(row) > 7 else "N/A"
        act_rows_str = row[2].strip() if len(row) > 2 else "0"

        mem_val = parse_memory_bytes(memory_str)
        if mem_val:
            total_memory += mem_val

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        if op_name in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
            m = re.search(r'data_scanned_rows:(\d+)', exec_info)
            if m:
                total_disk += float(m.group(1))

        if op_name in ("ExchangeSender", "ExchangeReceiver"):
            try:
                total_network += float(act_rows_str)
            except ValueError:
                pass

    cpu_time = parse_time_ms(rows[0][5].strip() if len(rows[0]) > 5 else "")

    return {
        "cpu_time_ms": max(cpu_time, 0),
        "memory_bytes": total_memory,
        "disk_io_rows": total_disk,
        "network_rows": total_network,
    }


def normalize_labels(labels: List[Dict]) -> Tuple[List[Dict], Dict]:
    keys = ["cpu_time_ms", "memory_bytes", "disk_io_rows", "network_rows"]
    log_labels = [{k: math.log(1.0 + max(l[k], 0)) for k in keys} for l in labels]
    stats = {}
    for k in keys:
        vals = [l[k] for l in log_labels]
        stats[k] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-8)}
    normalized = [{k: (l[k] - stats[k]["mean"]) / stats[k]["std"] for k in keys} for l in log_labels]
    return normalized, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_plans')
    parser.add_argument('--analyze-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_analyze_results')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--node-loss-weight', type=float, default=0.3,
                        help='Weight of per-node memory auxiliary loss')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Loading dataset with per-node memory labels...")
    graphs, query_labels_raw, meta = load_dataset_with_node_mem(args.plan_dir, args.analyze_dir)
    print(f"Loaded {len(graphs)} aligned plan-label pairs")

    # Count nodes with memory
    n_with_mem = sum(g.node_mem_mask.sum().item() for g in graphs)
    n_total_nodes = sum(g.x.shape[0] for g in graphs)
    print(f"Nodes with memory label: {n_with_mem}/{n_total_nodes} ({n_with_mem/n_total_nodes*100:.1f}%)")

    # Normalize query labels and node memory
    norm_labels, label_stats = normalize_labels(query_labels_raw)

    # Normalize node memory in log space
    all_node_mem = []
    for g in graphs:
        mask = g.node_mem_mask
        if mask.any():
            all_node_mem.extend(g.node_mem_label[mask].tolist())
    node_mem_mean = np.mean(all_node_mem)
    node_mem_std = max(np.std(all_node_mem), 1e-8)
    print(f"Node memory: mean(log)={node_mem_mean:.3f} std={node_mem_std:.3f}")

    # Normalize node memory labels
    for g in graphs:
        mask = g.node_mem_mask
        if mask.any():
            g.node_mem_label[mask] = (g.node_mem_label[mask] - node_mem_mean) / node_mem_std

    # Attach query labels
    key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_rows': 'net', 'cpu_time_ms': 'cpu'}
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in key_map.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    # Split
    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.6)]
    val_idx = indices[int(n * 0.6):int(n * 0.8)]
    test_idx = indices[int(n * 0.8):]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64)

    # Model
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    best_state = None
    w = args.node_loss_weight

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            preds = model(data)

            # Query-level loss
            qloss = sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                        for k in key_map.values())

            # Node-level memory auxiliary loss
            nloss = torch.tensor(0.0)
            if hasattr(data, 'node_mem_mask') and data.node_mem_mask.any():
                valid_mask = data.node_mem_mask
                node_pred = preds['node_mem'].squeeze(-1)[valid_mask]
                node_label = data.node_mem_label[valid_mask]
                nloss = F.huber_loss(node_pred, node_label)

            loss = qloss + w * nloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for data in val_loader:
                preds = model(data)
                qloss = sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                            for k in key_map.values()).item()
                val_loss += qloss
                n_b += 1
        val_loss /= max(n_b, 1)

        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f'E{epoch:3d} val={val_loss:.4f} best={best_val:.4f}')

    # Save & evaluate
    os.makedirs('/home/anqian/Desktop/my_lab/workloads/checkpoints', exist_ok=True)
    if best_state is not None:
        torch.save(best_state, '/home/anqian/Desktop/my_lab/workloads/checkpoints/best_sqlstorm_v2.pt')
        model.load_state_dict(best_state)
    else:
        print("WARNING: No valid best state found, using final model")
    model.eval()

    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64)
    all_p, all_t = {k: [] for k in key_map.values()}, {k: [] for k in key_map.values()}
    with torch.no_grad():
        for data in test_loader:
            preds = model(data)
            for k in key_map.values():
                all_p[k].append(preds[k].squeeze(-1).numpy())
                all_t[k].append(getattr(data, f'y_{k}').squeeze(-1).numpy())

    rmap = {'mem': 'memory_bytes', 'disk': 'disk_io_rows', 'net': 'network_rows', 'cpu': 'cpu_time_ms'}
    old = {'mem': (1.56, 11.41, 86.07, 0.78), 'disk': (1.18, 1.93, 2.44, 0.92),
           'net': (1.58, 3.93, 5.19, 0.95), 'cpu': (1.71, 4.02, 6.01, 0.59)}

    print(f"\n{'Dim':>8s}  {'P50':>8s} {'Δ':>6s}  {'P90':>8s} {'Δ':>6s}  {'P95':>8s} {'Δ':>6s}  {'R²':>8s} {'Δ':>6s}")
    print("-" * 78)
    for k, label in [('mem', 'Memory'), ('disk', 'DiskIO'), ('net', 'Network'), ('cpu', 'CPU')]:
        p = np.concatenate(all_p[k]).flatten()
        t = np.concatenate(all_t[k]).flatten()
        std, mn = label_stats[rmap[k]]['std'], label_stats[rmap[k]]['mean']
        p_raw = np.maximum(np.exp(p * std + mn) - 1, 0)
        t_raw = np.exp(t * std + mn) - 1
        qe = np.maximum(p_raw / np.maximum(t_raw, 1), np.maximum(t_raw, 1) / np.maximum(p_raw, 1))
        qs = np.sort(qe)
        nq = len(qs)
        r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
        print(f'{label:>8s}  {qs[nq//2]:>8.2f} {qs[nq//2]-old[k][0]:>+6.2f}  '
              f'{qs[int(nq*0.9)]:>8.2f} {qs[int(nq*0.9)]-old[k][1]:>+6.2f}  '
              f'{qs[int(nq*0.95)]:>8.2f} {qs[int(nq*0.95)]-old[k][2]:>+6.2f}  '
              f'{r2:>8.4f} {r2-old[k][3]:>+6.4f}')

    print(f"\nNode aux weight: {w}")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
