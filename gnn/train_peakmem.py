"""
Train PlanGNN with peak-path memory labels + NDV features.

Key change from previous version:
  Memory label = MAX(root-to-leaf path sum of per-operator memory)
  (not SUM of all operators)

This reflects real peak memory: operators execute bottom-up, children finish
and release memory before parents run.
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
OPERATOR_CLASS_MAP = {
    "TableFullScan": 0, "TableRangeScan": 0, "IndexRangeScan": 0,
    "TableRowIDScan": 0, "IndexLookUp": 0, "IndexReader": 0,
    "HashJoin": 1, "IndexHashJoin": 1, "IndexJoin": 1, "MergeJoin": 1,
    "HashAgg": 2, "StreamAgg": 2,
    "ExchangeSender": 3, "ExchangeReceiver": 3,
    "Sort": 4, "TopN": 4, "Window": 4,
    "Projection": 5, "Selection": 5, "CTEFullScan": 3,
}
LOCATION_MAP = {"root": 0, "mpp[tiflash]": 1, "cop[tikv]": 2, "tiflash": 1}
JOIN_TYPE_MAP = {"inner": 0, "anti": 1, "semi": 2, "left": 3, "right": 4, "none": 5}
EXCHANGE_TYPE_MAP = {"HashPartition": 0, "Broadcast": 1, "PassThrough": 2, "none": 3}


def load_ndv_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_join_columns(op_info: str) -> list:
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info)]


def parse_group_columns(op_info: str) -> list:
    if 'group by:' not in op_info: return []
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    if not m: return []
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', m.group(1))]


def parse_memory_bytes(raw: str) -> Optional[float]:
    raw = raw.strip()
    if raw.upper() == 'N/A' or raw == '':
        return None
    m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB|TB)', raw, re.I)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        return val * {"BYTES": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824, "TB": 1099511627776}[unit]
    return None


def compute_peak_path_memory(plan_text: str, analyze_text: str) -> float:
    """
    Compute maximum root-to-leaf path sum of per-operator memory.
    Returns peak memory in bytes.
    """
    # ─── Parse EXPLAIN plan to get tree structure ───
    plan_lines = []
    for line in plan_text.strip().split('\n'):
        if line.startswith('--') or not line.strip(): continue
        if '\t' in line: plan_lines.append(line)

    # ─── Parse EXPLAIN ANALYZE to get per-operator memory ───
    analyze_lines = analyze_text.strip().split('\n')
    node_mem = {}  # line_index → memory_bytes (or None)
    in_table = False
    for line in analyze_lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8:
                mem_val = parse_memory_bytes(parts[7].strip())
                if mem_val is not None:
                    node_mem[line] = mem_val  # Use raw line as key for matching
                else:
                    node_mem[line] = 0.0  # N/A → 0 for path computation

    # ─── Build tree: each node has depth, parent ───
    depths = []
    for line in plan_lines:
        stripped = line.lstrip(' │├└─')
        depths.append((len(line) - len(stripped)) // 2)

    # Find parent for each node
    parents = [-1] * len(plan_lines)
    indent_stack = []
    for idx, depth in enumerate(depths):
        while indent_stack and indent_stack[-1][0] >= depth:
            indent_stack.pop()
        if indent_stack:
            parents[idx] = indent_stack[-1][1]
        indent_stack.append((depth, idx))

    # ─── Get per-node memory from ANALYZE ───
    # Match plan lines to ANALYZE lines by operator ID
    mem_values = [0.0] * len(plan_lines)
    for idx, plan_line in enumerate(plan_lines):
        # Extract operator ID from plan line
        stripped = plan_line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) > 0:
            op_id = parts[0].strip()
        else:
            op_id = ""
        # Try to find matching ANALYZE line
        for aline, mem in node_mem.items():
            if op_id in aline.split('\t')[0]:
                mem_values[idx] = mem if mem is not None else 0.0
                break

    # ─── Compute peak path: max root-to-leaf sum ───
    children = defaultdict(list)
    for idx, p in enumerate(parents):
        if p >= 0:
            children[p].append(idx)

    roots = [i for i in range(len(plan_lines)) if parents[i] < 0]

    def max_path_sum(node):
        child_max = max((max_path_sum(c) for c in children[node]), default=0.0)
        return mem_values[node] + child_max

    if roots:
        return max(max_path_sum(r) for r in roots)
    return 0.0


def compute_ndv_features(plan_lines: list, ndv_cache: dict) -> list:
    """Compute (join_mem_log, agg_mem_log, sort_mem_log) per node."""
    results = []
    for line in plan_lines:
        stripped = line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) < 5:
            results.append((0.0, 0.0, 0.0)); continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip(); op_info = parts[4].strip()
        try: est_rows = max(float(est_rows_str), 1.0)
        except: est_rows = 1.0

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)

        jm, am, sm = 0.0, 0.0, 0.0

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
                jm = min(est_rows, min_ndv) * avg_width * (1.0 + 0.1 * n_equi)

        elif op_name in ('HashAgg', 'StreamAgg'):
            group_cols = parse_group_columns(op_info)
            if group_cols:
                ndvs = [ndv_cache.get(col, {"ndv": est_rows, "avg_width": 8}).get("ndv", est_rows) for col in group_cols]
                agg_ndv = min(ndvs) if ndvs else est_rows
                am = min(est_rows, agg_ndv) * 8 * len(group_cols)

        elif op_name in ('Sort', 'TopN'):
            n_sort = max(len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info)), 1)
            sm = est_rows * n_sort * 8

        results.append((math.log(1.0 + jm), math.log(1.0 + am), math.log(1.0 + sm)))
    return results


def parse_plan(plan_lines: list, ndv_feats: list) -> Optional[Data]:
    """Parse plan lines into PyG Data with NDV features."""
    nodes, indent_stack = [], []
    for idx, line in enumerate(plan_lines):
        stripped = line.lstrip(' │├└─')
        depth = (len(line) - len(stripped)) // 2
        parts = stripped.split('\t')
        if len(parts) < 3: continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip(); location_str = parts[2].strip()
        op_info = parts[4].strip() if len(parts) > 4 else ""

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        op_class = OPERATOR_CLASS_MAP.get(op_name, 5)

        try: est_rows = float(est_rows_str)
        except: est_rows = 1.0

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

    # Edges
    edges, indent_stack = [], []
    for idx, node in enumerate(nodes):
        depth = node["depth"]
        while indent_stack and indent_stack[-1][0] >= depth: indent_stack.pop()
        if indent_stack: edges.append((indent_stack[-1][1], idx))
        indent_stack.append((depth, idx))

    child_count = [0] * len(nodes)
    for p, c in edges: child_count[p] += 1
    for i, n in enumerate(nodes): n["children_count"] = child_count[i]

    children_map = defaultdict(list)
    for p, c in edges: children_map[p].append(c)

    def compute_subtree(i):
        s = nodes[i]["est_rows"]
        for c in children_map[i]: s += compute_subtree(c)
        nodes[i]["subtree_est"] = s; return s

    has_parent_set = {c for _, c in edges}
    for i in range(len(nodes)):
        if i not in has_parent_set: compute_subtree(i)

    max_depth = max(n["depth"] for n in nodes) if nodes else 1
    for n in nodes: n["depth_ratio"] = n["depth"] / max(max_depth, 1)

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
        ei = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        ea = []
        for p, c in edges:
            pa, ch = nodes[p], nodes[c]
            r = ch["est_rows"] / max(pa["est_rows"], 1.0) if pa["op_class"] == 1 else 1.0
            lp = pa["location_id"] * 3 + ch["location_id"]
            ea.append([r, float(lp), float(ch["exchange_type"]), float(ch["is_build"])])
        edge_attr = torch.tensor(ea, dtype=torch.float32)
    else:
        ei, edge_attr = torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4)

    rm = torch.zeros(len(nodes), dtype=torch.bool)
    for i in range(len(nodes)):
        if i not in has_parent_set: rm[i] = True; break

    return Data(x=x, edge_index=ei, edge_attr=edge_attr, root_mask=rm, n_nodes=len(nodes))


def parse_time_ms(raw: str) -> float:
    if not raw: return 0.0
    m = re.search(r'time:(\d+)m([\d.]+)s', raw)
    if m: return (int(m.group(1)) * 60 + float(m.group(2))) * 1000
    m = re.search(r'time:([\d.]+)(s|ms)', raw)
    if m: return float(m.group(1)) * 1000 if m.group(2) == 's' else float(m.group(1))
    return 0.0


def parse_query_labels(analyze_text: str) -> Optional[Dict]:
    """Extract query-level labels (disk, network, cpu) + peak path memory."""
    lines = analyze_text.strip().split('\n')
    rows = []
    in_table = False
    for line in lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line and not line.startswith('--'):
            parts = line.split('\t')
            if len(parts) >= 8: rows.append(parts)
    if not rows: return None

    total_disk, total_net = 0.0, 0.0
    for row in rows:
        raw_id = row[0].strip()
        exec_info = row[5].strip() if len(row) > 5 else ""
        act_rows_str = row[2].strip() if len(row) > 2 else "0"

        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)
        if op_name in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
            m = re.search(r'data_scanned_rows:(\d+)', exec_info)
            if m: total_disk += float(m.group(1))
        if op_name in ("ExchangeSender", "ExchangeReceiver"):
            try: total_net += float(act_rows_str)
            except: pass

    cpu_time = parse_time_ms(rows[0][5].strip() if rows and len(rows[0]) > 5 else "")
    return {"cpu_time_ms": max(cpu_time, 0), "disk_io_rows": total_disk, "network_rows": total_net}


def load_dataset(plan_dir: str, analyze_dir: str, ndv_cache: dict):
    """Load all aligned plan-label pairs with peak-path memory + NDV features."""
    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    graphs, labels, meta = [], [], []
    for fname in common:
        with open(os.path.join(plan_dir, fname)) as f: plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f: analyze_text = f.read()

        plan_lines = [l for l in plan_text.strip().split('\n') if '\t' in l and not l.startswith('--')]
        if not plan_lines: continue

        ndv_feats = compute_ndv_features(plan_lines, ndv_cache)
        g = parse_plan(plan_lines, ndv_feats)
        if g is None: continue

        # Query-level labels (disk, net, cpu)
        ql = parse_query_labels(analyze_text)
        if ql is None: continue

        # Peak path memory
        peak_mem = compute_peak_path_memory(plan_text, analyze_text)
        ql["memory_bytes"] = peak_mem

        graphs.append(g); labels.append(ql); meta.append(fname.replace('.txt', ''))

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

    print("Loading dataset with peak-path memory + NDV features...")
    graphs, labels, meta = load_dataset(args.plan_dir, args.analyze_dir, ndv_cache)
    print(f"  {len(graphs)} aligned plan-label pairs")

    # Compare old (sum) vs new (peak) memory labels
    import subprocess, json as _json
    old_mem = []
    for g, l, m in zip(graphs, labels, meta):
        old_mem.append(l["memory_bytes"])

    print(f"  Peak memory range: {min(old_mem):.0f}B ~ {max(old_mem)/1e9:.2f}GB")
    print(f"  Median peak memory: {np.median(old_mem)/1e3:.1f}KB")

    norm_labels, stats = normalize_labels(labels)
    for k, v in stats.items():
        print(f"  {k}: mean={v['mean']:.3f} std={v['std']:.3f}")

    key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_rows': 'net', 'cpu_time_ms': 'cpu'}
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in key_map.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.6)]
    val_idx = indices[int(n * 0.6):int(n * 0.8)]
    test_idx = indices[int(n * 0.8):]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64)

    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
    # Patch global_skip for NDV (13 inputs: 10 + 3 NDV)
    os_skip = model.global_skip
    new_skip = torch.nn.Sequential(
        torch.nn.Linear(13, 128), torch.nn.LeakyReLU(0.1), torch.nn.Linear(128, 128))
    new_skip[0].weight.data[:, :10] = os_skip[0].weight.data
    new_skip[0].bias.data = os_skip[0].bias.data
    new_skip[2].weight.data = os_skip[2].weight.data
    new_skip[2].bias.data = os_skip[2].bias.data
    model.global_skip = new_skip

    def ndv_forward(self, data):
        x = self._encode_nodes(data.x); x = self.node_encoder(x)
        e = self._encode_edges(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + conv(x, data.edge_index, edge_attr=e))
        from torch_geometric.nn import global_add_pool, global_max_pool
        h_max = global_max_pool(x, data.batch)
        gl = self.gate_mlp(x).squeeze(-1)
        hg = torch.stack([(torch.softmax(gl[data.batch == g], dim=0).unsqueeze(0) @
                           x[data.batch == g]).squeeze(0)
                          for g in range(int(data.batch.max()) + 1)])
        hs = global_add_pool(x, data.batch)
        plan_emb = self.out_proj(h_max + hg + hs)
        ns = torch.cat([data.x[:, 1:2], data.x[:, 3:4], data.x[:, 4:5], data.x[:, 5:6],
                        data.x[:, 6:7], data.x[:, 7:8], data.x[:, 8:9], data.x[:, 9:10],
                        data.x[:, 10:11]], dim=-1)
        gs = global_add_pool(ns, data.batch)
        nd = global_add_pool(data.x[:, 13:16], data.batch)
        nn = torch.bincount(data.batch + 1)[1:].float().unsqueeze(1)
        gf = self.global_skip(torch.cat([gs, nn, nd], dim=-1))
        pa = torch.cat([plan_emb, gf], dim=-1)
        return {"mem": self.mem_head(pa), "disk": self.disk_head(pa),
                "net": self.net_head(pa), "cpu": self.cpu_head(pa), "plan_emb": plan_emb}

    model.forward = ndv_forward.__get__(model, PlanGNN)
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
        val_loss, n_b = 0.0, 0
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
    torch.save(best_state, '/home/anqian/Desktop/my_lab/workloads/checkpoints/best_peakmem.pt')
    model.load_state_dict(best_state)
    model.eval()

    # Test evaluation
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64)
    all_p, all_t = {k: [] for k in key_map.values()}, {k: [] for k in key_map.values()}
    with torch.no_grad():
        for data in test_loader:
            preds = model(data)
            for k in key_map.values():
                all_p[k].append(preds[k].squeeze(-1).numpy())
                all_t[k].append(getattr(data, f'y_{k}').squeeze(-1).numpy())

    rmap = {'mem': 'memory_bytes', 'disk': 'disk_io_rows', 'net': 'network_rows', 'cpu': 'cpu_time_ms'}

    print(f"\n{'Dim':>8s}  {'P50':>8s}  {'P80':>8s}  {'P90':>8s}  {'P95':>8s}  {'R²':>8s}")
    print("-" * 58)
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
        print(f'{label:>8s}  {qs[nq // 2]:>8.2f}  {qs[int(nq * 0.8)]:>8.2f}  '
              f'{qs[int(nq * 0.9)]:>8.2f}  {qs[int(nq * 0.95)]:>8.2f}  {r2:>8.4f}')

    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
