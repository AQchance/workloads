"""
Train PlanGNN with NDV-based memory features.

Adds 3 per-node features from column statistics:
  - join_mem_log: log(1 + estimated_hash_table_bytes) for JOIN nodes
  - agg_mem_log:  log(1 + estimated_agg_memory) for AGG nodes
  - sort_mem_log: log(1 + estimated_sort_memory) for SORT nodes

These features are computed at parse time from TiDB column NDV statistics.
Total node features: 13 (original) + 3 (NDV) = 16
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
NODE_RAW_DIM = 16  # 13 original + 3 NDV-based

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


def load_ndv_cache(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Convert JSON keys back; JSON doesn't have native defaultdict-like behavior
    return raw


def parse_join_columns(op_info: str) -> list:
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info)]


def parse_group_columns(op_info: str) -> list:
    if 'group by:' not in op_info:
        return []
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    if not m:
        return []
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', m.group(1))]


def compute_ndv_features(plan_text: str, ndv_cache: dict) -> list:
    """Compute per-node (join_mem_log, agg_mem_log, sort_mem_log)."""
    lines = [l for l in plan_text.strip().split('\n') if '\t' in l and not l.startswith('--')]
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
        is_build = '(Build)' in raw_id

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


def parse_plan(plan_text: str, ndv_cache: dict) -> Optional[Data]:
    """Parse EXPLAIN plan into PyG Data with NDV features."""
    lines = [l for l in plan_text.strip().split('\n') if '\t' in l and not l.startswith('--')]
    if not lines:
        return None

    ndv_feats = compute_ndv_features(plan_text, ndv_cache)
    assert len(ndv_feats) == len(lines), f"NDV feat count mismatch: {len(ndv_feats)} vs {len(lines)}"

    nodes, indent_stack, node_raw_ids = [], [], []
    for idx, line in enumerate(lines):
        stripped = line.lstrip(' │├└─')
        depth = (len(line) - len(stripped)) // 2
        parts = stripped.split('\t')
        if len(parts) < 3:
            continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip()
        location_str = parts[2].strip()
        op_info = parts[4].strip() if len(parts) > 4 else ""

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        op_class = OPERATOR_CLASS_MAP.get(op_name, 5)

        try: est_rows = float(est_rows_str)
        except ValueError: est_rows = 1.0

        loc_id = LOCATION_MAP.get(location_str, 0)
        stream = int(re.search(r'stream_count:\s*(\d+)', op_info).group(1)) if 'stream_count' in op_info else 1
        jt = JOIN_TYPE_MAP.get((re.search(r'(inner|anti|semi|left|right)\s+join', op_info, re.I) or [None, 'none'])[1].lower(), 5)
        et = EXCHANGE_TYPE_MAP.get((re.search(r'ExchangeType:\s*(\w+)', op_info) or [None, 'none'])[1], 3)
        ne = len(re.findall(r'eq\(', op_info))
        ng = (len(re.search(r'group by:(.*?)(?:, funcs:|$)', op_info).group(1).split(',')) if 'group by:' in op_info else 0)
        ns = len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)) if op_class == 4 else 0
        hf = 1 if re.search(r'pushed down filter:(?!\s*empty)', op_info) else 0
        ib = 1 if '(Build)' in raw_id else 0

        jml, aml, sml = ndv_feats[idx]

        nodes.append({
            "depth": depth, "op_class": op_class, "est_rows": est_rows,
            "location_id": loc_id, "stream_count": stream,
            "join_type": jt, "exchange_type": et,
            "n_equi": ne, "n_group": ng, "n_sort": ns,
            "has_filter": hf, "is_build": ib,
            "join_mem_log": jml, "agg_mem_log": aml, "sort_mem_log": sml,
        })
        node_raw_ids.append(raw_id)

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

    # Feature matrix [N, 16]
    x_list = []
    for n in nodes:
        x_list.append([
            float(n["op_class"]), math.log(1.0 + n["est_rows"]),
            float(n["location_id"]), float(n["stream_count"]),
            float(n["children_count"]), n["depth_ratio"],
            float(n["n_equi"]), float(n["n_group"]), float(n["n_sort"]),
            float(n["has_filter"]), math.log(1.0 + n["subtree_est"]),
            float(n["join_type"]), float(n["exchange_type"]),
            n["join_mem_log"], n["agg_mem_log"], n["sort_mem_log"],
        ])
    x = torch.tensor(x_list, dtype=torch.float32)

    if edges:
        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        e_attr_list = []
        for p, c in edges:
            parent, child = nodes[p], nodes[c]
            ratio = child["est_rows"] / max(parent["est_rows"], 1.0) if parent["op_class"] == 1 else 1.0
            loc_pair = parent["location_id"] * 3 + child["location_id"]
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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, root_mask=root_mask, n_nodes=len(nodes))


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


def parse_analyze(text: str) -> Optional[Dict]:
    """Extract query-level resource labels."""
    lines = text.strip().split('\n')
    rows = []
    in_table = False
    for line in lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8: rows.append(parts)
    if not rows: return None

    total_mem, total_disk, total_net = 0.0, 0.0, 0.0
    for row in rows:
        raw_id = row[0].strip()
        exec_info = row[5].strip() if len(row) > 5 else ""
        memory_str = row[7].strip() if len(row) > 7 else "N/A"
        act_rows_str = row[2].strip() if len(row) > 2 else "0"

        mem_val = parse_memory_bytes(memory_str)
        if mem_val: total_mem += mem_val

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        if op_name in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
            m = re.search(r'data_scanned_rows:(\d+)', exec_info)
            if m: total_disk += float(m.group(1))
        if op_name in ("ExchangeSender", "ExchangeReceiver"):
            try: total_net += float(act_rows_str)
            except ValueError: pass

    cpu_time = parse_time_ms(rows[0][5].strip() if rows and len(rows[0]) > 5 else "")
    return {"cpu_time_ms": max(cpu_time, 0), "memory_bytes": total_mem,
            "disk_io_rows": total_disk, "network_rows": total_net}


def load_dataset(plan_dir: str, analyze_dir: str, ndv_cache: dict):
    """Load all aligned plan-label pairs with NDV features."""
    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    graphs, labels, meta = [], [], []
    for fname in common:
        with open(os.path.join(plan_dir, fname)) as f: plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f: analyze_text = f.read()

        g = parse_plan(plan_text, ndv_cache)
        lab = parse_analyze(analyze_text)
        if g is None or lab is None or g.x.shape[0] == 0:
            continue
        graphs.append(g)
        labels.append(lab)
        meta.append(fname.replace('.txt', ''))

    return graphs, labels, meta


def normalize_labels(labels: List[Dict]) -> Tuple[List[Dict], Dict]:
    keys = ["cpu_time_ms", "memory_bytes", "disk_io_rows", "network_rows"]
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
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Loading NDV cache...")
    ndv_cache = load_ndv_cache(args.ndv_cache)
    print(f"  {len(ndv_cache)} columns")

    print("Loading dataset with NDV features...")
    graphs, labels, meta = load_dataset(args.plan_dir, args.analyze_dir, ndv_cache)
    print(f"  {len(graphs)} aligned plan-label pairs")

    norm_labels, stats = normalize_labels(labels)
    key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_rows': 'net', 'cpu_time_ms': 'cpu'}
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

    # Model with adjusted feature dims
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
    # Patch the node encoder for 16-dim raw features
    # We need to update _encode_nodes to handle the new NDV features
    # Quick patch: modify the model's internal structure
    # The _encode_nodes currently expects 13 dims; we have 16.
    # We'll monkey-patch for now.
    import model as mdl
    original_encode = model._encode_nodes

    def patched_encode(self, x):
        # Original categorical embeddings (56 dims)
        op_class_id = x[:, 0].long()
        location_id = x[:, 2].long()
        join_type_id = x[:, 11].long()
        exchange_type_id = x[:, 12].long()
        cat_emb = torch.cat([
            self.op_class_emb(op_class_id),
            self.location_emb(location_id),
            self.join_type_emb(join_type_id),
            self.exchange_type_emb(exchange_type_id),
        ], dim=-1)  # 56

        # Original scalars (9 dims)
        orig_scalars = torch.cat([
            x[:, 1:2], x[:, 3:4], x[:, 4:5], x[:, 5:6],
            x[:, 6:7], x[:, 7:8], x[:, 8:9], x[:, 9:10], x[:, 10:11],
        ], dim=-1)
        scalar_proj = self.scalar_proj(orig_scalars)  # 9 → 32

        # New NDV scalars (3 dims) — add directly to the encoder input
        ndv_feats = x[:, 13:16]  # join_mem_log, agg_mem_log, sort_mem_log

        return torch.cat([cat_emb, scalar_proj, ndv_feats], dim=-1)  # 56 + 32 + 3 = 91

    model._encode_nodes = patched_encode.__get__(model, PlanGNN)

    # Also patch the node_encoder for the new input dim: 91
    old_enc = model.node_encoder
    new_first = torch.nn.Linear(91, 128)
    new_first.weight.data[:, :88] = old_enc[0].weight.data
    new_first.bias.data = old_enc[0].bias.data
    old_enc[0] = new_first
    model.node_encoder = old_enc

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            preds = model(data)
            loss = sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                        for k in key_map.values())
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
                val_loss += sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                                for k in key_map.values()).item()
                n_b += 1
        val_loss /= max(n_b, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f'E{epoch:3d} val={val_loss:.4f} best={best_val:.4f}')

    os.makedirs('/home/anqian/Desktop/my_lab/workloads/checkpoints', exist_ok=True)
    torch.save(best_state, '/home/anqian/Desktop/my_lab/workloads/checkpoints/best_ndv.pt')
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
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
        std, mn = stats[rmap[k]]['std'], stats[rmap[k]]['mean']
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

    print(f"\nNDV features enabled: +join_mem_log, +agg_mem_log, +sort_mem_log per node")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
