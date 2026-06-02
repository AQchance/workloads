"""
Train PlanGNN on SQLStorm queries with TAB-separated EXPLAIN plans.

Usage: python train_sqlstorm.py --epochs 200
"""

import os, sys, re, math, argparse, json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN

# ─── Operator class mapping ───
OPERATOR_CLASS_MAP = {
    "TableFullScan": 0, "TableRangeScan": 0, "IndexRangeScan": 0,
    "TableRowIDScan": 0, "IndexLookUp": 0, "IndexReader": 0,
    "HashJoin": 1, "IndexHashJoin": 1, "IndexJoin": 1, "MergeJoin": 1,
    "HashAgg": 2, "StreamAgg": 2,
    "ExchangeSender": 3, "ExchangeReceiver": 3,
    "Sort": 4, "TopN": 4,
    "Projection": 5, "Selection": 5, "Window": 4,  # Window ~ Sort-like
}
LOCATION_MAP = {"root": 0, "mpp[tiflash]": 1, "cop[tikv]": 2, "tiflash": 1}
JOIN_TYPE_MAP = {"inner": 0, "anti": 1, "semi": 2, "left": 3, "right": 4, "none": 5}
EXCHANGE_TYPE_MAP = {"HashPartition": 0, "Broadcast": 1, "PassThrough": 2, "none": 3}

N_OP_CLASSES, N_LOCATIONS = 6, 3
N_JOIN_TYPES, N_EXCHANGE_TYPES = 6, 4
NODE_RAW_DIM = 13  # Same as before


def parse_tab_plan(text: str) -> Data:
    """Parse a TAB-separated EXPLAIN plan into a PyG Data object."""
    lines = text.strip().split('\n')
    plan_lines = []
    for line in lines:
        if line.startswith('--') or not line.strip():
            continue
        if '\t' in line:
            plan_lines.append(line)

    if not plan_lines:
        return None

    nodes = []
    indent_stack = []

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

        # Normalize operator name
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        op_class = OPERATOR_CLASS_MAP.get(op_name, 5)

        # Parse estRows
        try:
            est_rows = float(est_rows_str)
        except ValueError:
            est_rows = 1.0

        # Location
        loc_id = LOCATION_MAP.get(location_str, 0)

        # Parse op_info
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
            "depth": depth,
            "op_class": op_class, "est_rows": est_rows, "location_id": loc_id,
            "stream_count": stream_count, "join_type": join_type,
            "exchange_type": exchange_type, "n_equi": n_equi,
            "n_group": n_group, "n_sort": n_sort,
            "has_filter": has_filter, "is_build": is_build,
        })

    # Build edges from indent stack
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

    # Subtree estRows (post-order)
    children_map = defaultdict(list)
    for p, c in edges:
        children_map[p].append(c)

    def compute_subtree(i):
        s = nodes[i]["est_rows"]
        for c in children_map[i]:
            s += compute_subtree(c)
        nodes[i]["subtree_est"] = s
        return s

    has_parent = {c for _, c in edges}
    for i in range(len(nodes)):
        if i not in has_parent:
            compute_subtree(i)

    max_depth = max(n["depth"] for n in nodes) if nodes else 1
    for n in nodes:
        n["depth_ratio"] = n["depth"] / max(max_depth, 1)

    # Build feature matrix [N, 13]
    x_list = []
    for n in nodes:
        feats = [
            float(n["op_class"]),
            math.log(1.0 + n["est_rows"]),
            float(n["location_id"]),
            float(n["stream_count"]),
            float(n["children_count"]),
            n["depth_ratio"],
            float(n["n_equi"]),
            float(n["n_group"]),
            float(n["n_sort"]),
            float(n["has_filter"]),
            math.log(1.0 + n["subtree_est"]),
            float(n["join_type"]),
            float(n["exchange_type"]),
        ]
        x_list.append(feats)

    x = torch.tensor(x_list, dtype=torch.float32)

    # Edges
    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        e_attr_list = []
        for p, c in edges:
            parent, child = nodes[p], nodes[c]
            ratio = child["est_rows"] / max(parent["est_rows"], 1.0) if parent["op_class"] == 1 else 1.0
            loc_pair = parent["location_id"] * N_LOCATIONS + child["location_id"]
            e_attr_list.append([ratio, float(loc_pair), float(child["exchange_type"]), float(child["is_build"])])
        edge_attr = torch.tensor(e_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4, dtype=torch.float32)

    root_mask = torch.zeros(len(nodes), dtype=torch.bool)
    for i in range(len(nodes)):
        if i not in has_parent:
            root_mask[i] = True
            break

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, root_mask=root_mask, n_nodes=len(nodes))


def parse_explain_analyze(text: str) -> Dict:
    """Extract resource labels from TAB-separated EXPLAIN ANALYZE output."""
    lines = text.strip().split('\n')

    # Find table rows (tab-separated, contain operator info)
    rows = []
    in_table = False
    for line in lines:
        if line.startswith('--') or not line.strip():
            continue
        if line.startswith('id\t'):
            in_table = True
            continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8:
                rows.append(parts)

    if not rows:
        return None

    # Extract labels
    total_memory = 0.0
    total_disk_rows = 0.0
    total_network_rows = 0.0

    for row in rows:
        raw_id = row[0].strip()
        exec_info = row[5].strip() if len(row) > 5 else ""
        memory_str = row[7].strip() if len(row) > 7 else "N/A"
        act_rows_str = row[2].strip() if len(row) > 2 else "0"

        # Memory
        if memory_str.upper() != 'N/A' and memory_str:
            m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB)', memory_str, re.I)
            if m:
                val, unit = float(m.group(1)), m.group(2).upper()
                total_memory += val * {"BYTES": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824}.get(unit, 1)

        # Disk: data_scanned_rows for scan operators
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        if op_name in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
            m = re.search(r'data_scanned_rows:(\d+)', exec_info)
            if m:
                total_disk_rows += float(m.group(1))

        # Network: actRows for exchange operators
        if op_name in ("ExchangeSender", "ExchangeReceiver"):
            try:
                total_network_rows += float(act_rows_str)
            except ValueError:
                pass

    # CPU time from root operator
    cpu_time_ms = 0.0
    if rows:
        root_exec = rows[0][5].strip() if len(rows[0]) > 5 else ""
        m = re.search(r'time:(\d+)m([\d.]+)s', root_exec)
        if m:
            cpu_time_ms = (int(m.group(1)) * 60 + float(m.group(2))) * 1000
        else:
            m = re.search(r'time:([\d.]+)(s|ms)', root_exec)
            if m:
                val, unit = float(m.group(1)), m.group(2)
                cpu_time_ms = val * 1000 if unit == 's' else val

    return {
        "cpu_time_ms": max(cpu_time_ms, 0),
        "memory_bytes": total_memory,
        "disk_io_rows": total_disk_rows,
        "network_rows": total_network_rows,
    }


def load_dataset(plan_dir: str, analyze_dir: str) -> Tuple[List[Data], List[Dict], List[str]]:
    """Load all aligned plan-label pairs."""
    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    graphs, labels, meta = [], [], []
    for fname in common:
        qnum = fname.replace('.txt', '')
        with open(os.path.join(plan_dir, fname)) as f:
            plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f:
            analyze_text = f.read()

        g = parse_tab_plan(plan_text)
        lab = parse_explain_analyze(analyze_text)
        if g is None or lab is None or g.x.shape[0] == 0:
            continue

        graphs.append(g)
        labels.append(lab)
        meta.append(qnum)

    return graphs, labels, meta


def normalize_labels(labels: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Log-transform + z-score per dimension."""
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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    graphs, labels, meta = load_dataset(args.plan_dir, args.analyze_dir)
    print(f"Loaded {len(graphs)} aligned plan-label pairs")

    # Template diversity stats
    n_nodes_set = set(g.x.shape[0] for g in graphs)
    n_edges_set = set(g.edge_index.shape[1] for g in graphs)
    uniq_shapes = set((g.x.shape[0], g.edge_index.shape[1]) for g in graphs)
    print(f"Unique plan shapes: {len(uniq_shapes)} (vs 21 in TPC-H)")

    norm_labels, label_stats = normalize_labels(labels)
    for k, v in label_stats.items():
        print(f"  {k}: mean={v['mean']:.3f} std={v['std']:.3f}")

    # Attach labels
    key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_rows': 'net', 'cpu_time_ms': 'cpu'}
    for g, nl in zip(graphs, norm_labels):
        for raw_k, short_k in key_map.items():
            setattr(g, f'y_{short_k}', torch.tensor([nl[raw_k]], dtype=torch.float32))

    # Train/val/test split: 60/20/20 random
    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.6)]
    val_idx = indices[int(n * 0.6):int(n * 0.8)]
    test_idx = indices[int(n * 0.8):]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=args.batch_size)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=args.batch_size)

    device = torch.device(args.device)
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.1).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            preds = model(data)
            loss = sum(F.huber_loss(preds[k], getattr(data, f'y_{k}')) for k in key_map.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                preds = model(data)
                val_loss += sum(F.huber_loss(preds[k], getattr(data, f'y_{k}')) for k in key_map.values()).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '/home/anqian/Desktop/my_lab/workloads/checkpoints/best_sqlstorm.pt')
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"E{epoch:4d} train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    # Test evaluation
    model.load_state_dict(torch.load('/home/anqian/Desktop/my_lab/workloads/checkpoints/best_sqlstorm.pt'))
    model.eval()

    # Oracle baseline: per-query prediction is the mean (but only 1 sample per template → no template grouping)
    # For diverse queries, the oracle = predicting global mean (not template mean)
    print(f"\n{'='*70}")
    print(f"Test results ({len(test_idx)} queries):")
    print(f"{'='*70}")

    test_preds, test_targets = {k: [] for k in key_map.values()}, {k: [] for k in key_map.values()}
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            preds = model(data)
            for k in key_map.values():
                test_preds[k].append(preds[k].cpu().numpy())
                test_targets[k].append(getattr(data, f'y_{k}').cpu().numpy())

    raw_key_map = {'mem': 'memory_bytes', 'disk': 'disk_io_rows', 'net': 'network_rows', 'cpu': 'cpu_time_ms'}
    for k, label in [('mem', 'Memory'), ('disk', 'DiskIO'), ('net', 'Network'), ('cpu', 'CPU')]:
        p = np.concatenate(test_preds[k]).flatten()
        t = np.concatenate(test_targets[k]).flatten()
        r2 = 1 - np.sum((t-p)**2) / max(np.sum((t-np.mean(t))**2), 1e-8)
        # Denormalize MAE
        std, mn = label_stats[raw_key_map[k]]['std'], label_stats[raw_key_map[k]]['mean']
        p_raw = np.exp(p * std + mn) - 1
        t_raw = np.exp(t * std + mn) - 1
        mae_raw = np.mean(np.abs(p_raw - t_raw))
        # Oracle (predict global mean)
        oracle_mse = np.mean((t - np.mean(t))**2)

        if k == 'mem': unit = f'{mae_raw/1e3:.0f}KB'
        elif k == 'cpu': unit = f'{mae_raw/1e3:.1f}s'
        else: unit = f'{mae_raw/1e6:.0f}Mrow'

        print(f"  {label:>8s}: R²={r2:.4f} MAE={unit:>12s} Oracle_MSE={oracle_mse:.4f}")

    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
