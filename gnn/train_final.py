"""
Final GNN training: unified normalization, NDV features in node encoder,
CPU/Memory classification, Disk regression, Network estRows linear regression.

Usage:
    cd /home/anqian/Desktop/my_lab/workloads
    python gnn/train_final.py --epochs 400

Output:
    checkpoints/best_final.pt  — best model weights
"""

import os, sys, re, math, json, argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN

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
N_CLASSES = 3


def load_ndv_cache(path):
    with open(path) as f: return json.load(f)


def parse_join_columns(op_info):
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info)]


def parse_group_columns(op_info):
    if 'group by:' not in op_info: return []
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    return [f"{t}.{c}" for t, c in re.findall(r'tpch_sf40\.(\w+)\.(\w+)', m.group(1))] if m else []


def extract_cpu_resource(analyze_text):
    total = 0.0; in_table = False
    for line in analyze_text.strip().split('\n'):
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line:
            parts = line.split('\t')
            if len(parts) < 6: continue
            ei = parts[5].strip()
            pm = re.search(r'proc max:([\d.]+)(s|ms)', ei)
            if pm:
                v = float(pm.group(1))
                if pm.group(2) == 'ms': v /= 1000
                th = re.search(r'threads:(\d+)', ei)
                ts = re.search(r'tasks:(\d+)', ei)
                total += v * (int(th.group(1)) if th else 1) * (int(ts.group(1)) if ts else 1)
            tp = re.search(r'tot_proc:([\d.]+)(s|ms)', ei)
            if tp:
                v = float(tp.group(1))
                if tp.group(2) == 'ms': v /= 1000; total += v
    return total


def parse_memory_bytes(raw):
    raw = raw.strip()
    if raw.upper() == 'N/A' or raw == '': return None
    m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB|TB)', raw, re.I)
    if m:
        v, u = float(m.group(1)), m.group(2).upper()
        return v * {"BYTES": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824, "TB": 1099511627776}[u]
    return None


def extract_labels(analyze_text):
    lines = analyze_text.strip().split('\n')
    tm, td, tn = 0.0, 0.0, 0.0; in_table = False
    for line in lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line:
            parts = line.split('\t')
            if len(parts) < 8: continue
            rid = parts[0].strip(); ei = parts[5].strip()
            ms = parts[7].strip(); ar = parts[2].strip()
            mv = parse_memory_bytes(ms)
            if mv: tm += mv
            on = re.sub(r'^[│├└─\s]+', '', rid)
            on = re.sub(r'\(Build\)|\(Probe\)', '', on).strip()
            on = re.sub(r'_\d+$', '', on)
            if on in ("TableFullScan", "TableRangeScan", "IndexRangeScan", "TableRowIDScan"):
                m = re.search(r'data_scanned_rows:(\d+)', ei)
                if m: td += float(m.group(1))
            if on in ("ExchangeSender", "ExchangeReceiver"):
                try: tn += float(ar)
                except: pass
    cpu = extract_cpu_resource(analyze_text)
    if cpu <= 0: return None
    return {"cpu_resource": cpu, "memory_bytes": tm, "disk_io_rows": td, "network_rows": tn}


def compute_ndv_features(plan_lines, ndv_cache):
    results = []
    for line in plan_lines:
        stripped = line.lstrip(' │├└─'); parts = stripped.split('\t')
        if len(parts) < 5: results.append((0.0, 0.0, 0.0)); continue
        rid = parts[0].strip(); es = parts[1].strip()
        oi = parts[4].strip() if len(parts) > 4 else ""
        try: er = max(float(es), 1.0)
        except: er = 1.0
        on = re.sub(r'^[│├└─\s]+', '', rid)
        on = re.sub(r'\(Build\)|\(Probe\)', '', on).strip()
        on = re.sub(r'_\d+$', '', on)
        jm = am = sm = 0.0
        if on in ('HashJoin', 'IndexHashJoin', 'IndexJoin', 'MergeJoin'):
            cols = parse_join_columns(oi)
            if cols:
                ndvs = [ndv_cache.get(col, {"ndv": er, "avg_width": 8}).get("ndv", er) for col in cols]
                ws = [ndv_cache.get(col, {"ndv": er, "avg_width": 8}).get("avg_width", 8) for col in cols]
                ne = len(re.findall(r'eq\(', oi))
                aw = sum(ws) / len(ws) if ws else 8
                jm = min(er, min(ndvs)) * aw * (1.0 + 0.1 * ne)
        elif on in ('HashAgg', 'StreamAgg'):
            cols = parse_group_columns(oi)
            if cols:
                ndvs = [ndv_cache.get(col, {"ndv": er, "avg_width": 8}).get("ndv", er) for col in cols]
                am = min(er, min(ndvs)) * 8 * len(cols)
        elif on in ('Sort', 'TopN'):
            ns = max(len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', oi)), 1)
            sm = er * ns * 8
        results.append((math.log(1.0 + jm), math.log(1.0 + am), math.log(1.0 + sm)))
    return results


def parse_plan(plan_lines, ndv_feats):
    nodes = []
    for idx, line in enumerate(plan_lines):
        stripped = line.lstrip(' │├└─')
        depth = (len(line) - len(stripped)) // 2
        parts = stripped.split('\t')
        if len(parts) < 3: continue
        rid = parts[0].strip(); es = parts[1].strip()
        loc = parts[2].strip(); oi = parts[4].strip() if len(parts) > 4 else ""
        on = re.sub(r'^[│├└─\s]+', '', rid)
        on = re.sub(r'\(Build\)|\(Probe\)', '', on).strip()
        on = re.sub(r'_\d+$', '', on)
        oc = OPERATOR_CLASS_MAP.get(on, 5)
        try: er = float(es)
        except: er = 1.0
        li = LOCATION_MAP.get(loc, 0)
        sc = int(re.search(r'stream_count:\s*(\d+)', oi).group(1)) if 'stream_count' in oi else 1
        jt = JOIN_TYPE_MAP.get((re.search(r'(inner|anti|semi|left|right)\s+join', oi, re.I) or [None, 'none'])[1].lower(), 5)
        et = EXCHANGE_TYPE_MAP.get((re.search(r'ExchangeType:\s*(\w+)', oi) or [None, 'none'])[1], 3)
        ne = len(re.findall(r'eq\(', oi))
        ng = (len(re.search(r'group by:(.*?)(?:, funcs:|$)', oi).group(1).split(',')) if 'group by:' in oi else 0)
        ns = len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', oi)) if oc == 4 else 0
        hf = 1 if re.search(r'pushed down filter:(?!\s*empty)', oi) else 0
        ib = 1 if '(Build)' in rid else 0
        jml, aml, sml = ndv_feats[idx]
        nodes.append({"depth": depth, "op_class": oc, "est_rows": er, "location_id": li,
                       "stream_count": sc, "join_type": jt, "exchange_type": et, "n_equi": ne,
                       "n_group": ng, "n_sort": ns, "has_filter": hf, "is_build": ib,
                       "join_mem_log": jml, "agg_mem_log": aml, "sort_mem_log": sml})

    edges, istack = [], []
    for idx, node in enumerate(nodes):
        while istack and istack[-1][0] >= node["depth"]: istack.pop()
        if istack: edges.append((istack[-1][1], idx))
        istack.append((node["depth"], idx))
    cc = [0] * len(nodes)
    for p, c in edges: cc[p] += 1
    for i, n in enumerate(nodes): n["children_count"] = cc[i]
    cmap = defaultdict(list)
    for p, c in edges: cmap[p].append(c)

    def sub(i):
        s = nodes[i]["est_rows"]
        for c in cmap[i]: s += sub(c)
        nodes[i]["subtree_est"] = s; return s

    hp = {c for _, c in edges}
    for i in range(len(nodes)):
        if i not in hp: sub(i)
    md = max(n["depth"] for n in nodes) if nodes else 1
    for n in nodes: n["depth_ratio"] = n["depth"] / max(md, 1)

    xl = [[float(n["op_class"]), math.log(1.0 + n["est_rows"]), float(n["location_id"]),
           float(n["stream_count"]), float(n["children_count"]), n["depth_ratio"],
           float(n["n_equi"]), float(n["n_group"]), float(n["n_sort"]),
           float(n["has_filter"]), math.log(1.0 + n["subtree_est"]),
           float(n["join_type"]), float(n["exchange_type"]),
           n["join_mem_log"], n["agg_mem_log"], n["sort_mem_log"]] for n in nodes]
    x = torch.tensor(xl, dtype=torch.float32)

    if edges:
        ei = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t().contiguous()
        ea = [[nodes[c]["est_rows"] / max(nodes[p]["est_rows"], 1.0) if nodes[p]["op_class"] == 1 else 1.0,
               float(nodes[p]["location_id"] * 3 + nodes[c]["location_id"]),
               float(nodes[c]["exchange_type"]), float(nodes[c]["is_build"])] for p, c in edges]
        edge_attr = torch.tensor(ea, dtype=torch.float32)
    else:
        ei, edge_attr = torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4)

    rm = torch.zeros(len(nodes), dtype=torch.bool)
    for i in range(len(nodes)):
        if i not in hp: rm[i] = True; break
    return Data(x=x, edge_index=ei, edge_attr=edge_attr, root_mask=rm, n_nodes=len(nodes))


# Use the proven parser from train_ndv.py
from train_ndv import load_dataset as _load_ndv_dataset
def load_dataset(plan_dir, analyze_dir, ndv_cache):
    return _load_ndv_dataset(plan_dir, analyze_dir, ndv_cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_plans')
    parser.add_argument('--analyze-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_analyze_results')
    parser.add_argument('--ndv-cache', default='/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ndv_cache = load_ndv_cache(args.ndv_cache)

    print("Loading data...")
    graphs, labels_raw, meta = load_dataset(args.plan_dir, args.analyze_dir, ndv_cache)
    print(f"  {len(graphs)} queries")

    # Extract CPU resource from ANALYZE files if not in labels
    cpu_raw = []
    for i, l in enumerate(labels_raw):
        if 'cpu_resource' in l:
            cpu_raw.append(l['cpu_resource'])
        else:
            with open(f"{args.analyze_dir}/{meta[i]}.txt") as f:
                cpu_raw.append(extract_cpu_resource(f.read()))

    cpu_log = np.log(1 + np.array(cpu_raw))
    mem_log = np.log(1 + np.array([l['memory_bytes'] for l in labels_raw]))
    disk_log = np.log(1 + np.array([l['disk_io_rows'] for l in labels_raw]))
    net_log = np.log(1 + np.array([l['network_rows'] for l in labels_raw]))

    # Split
    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.7)]
    val_idx = indices[int(n * 0.7):int(n * 0.85)]
    test_idx = indices[int(n * 0.85):]
    print(f"  Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ─── All stats from TRAINING SET ONLY ───
    disk_mean = np.mean(disk_log[train_idx]); disk_std = max(np.std(disk_log[train_idx]), 1e-8)
    disk_norm = (disk_log - disk_mean) / disk_std
    cpu_mean = np.mean(cpu_log[train_idx]); cpu_std = max(np.std(cpu_log[train_idx]), 1e-8)
    cpu_norm = (cpu_log - cpu_mean) / cpu_std

    cpu_p33 = np.percentile(cpu_log[train_idx], 33.33); cpu_p67 = np.percentile(cpu_log[train_idx], 66.67)
    mem_p33 = np.percentile(mem_log[train_idx], 33.33); mem_p67 = np.percentile(mem_log[train_idx], 66.67)
    cpu_cls = np.digitize(cpu_log, [cpu_p33, cpu_p67]); mem_cls = np.digitize(mem_log, [mem_p33, mem_p67])

    print(f"  CPU thresholds (log): {cpu_p33:.2f} / {cpu_p67:.2f}")
    print(f"  MEM thresholds (log): {mem_p33:.2f} / {mem_p67:.2f}")

    # Network: Exchange estRows for LR
    def exch_est(g):
        return sum(math.exp(g.x[i, 1].item()) - 1 for i in range(g.x.shape[0]) if int(g.x[i, 0].item()) == 3)

    exch_log = np.log(1 + np.array([exch_est(g) for g in graphs]))
    net_lr = LinearRegression().fit(exch_log[train_idx].reshape(-1, 1), net_log[train_idx])

    # Attach labels
    for i, g in enumerate(graphs):
        g.y_disk = torch.tensor([disk_norm[i]], dtype=torch.float32)
        g.y_cpu_reg = torch.tensor([cpu_norm[i]], dtype=torch.float32)
        g.y_cpu_cls = torch.tensor([cpu_cls[i]], dtype=torch.long)
        g.y_mem_cls = torch.tensor([mem_cls[i]], dtype=torch.long)

    # ─── Model ───
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)

    # NDV in node encoder (monkey-patch: 88 → 91 dims)
    orig_encode = model._encode_nodes

    def patched_encode(self, x):
        cat_emb = torch.cat([
            self.op_class_emb(x[:, 0].long()),
            self.location_emb(x[:, 2].long()),
            self.join_type_emb(x[:, 11].long()),
            self.exchange_type_emb(x[:, 12].long()),
        ], dim=-1)
        scalars = torch.cat([x[:, 1:2], x[:, 3:4], x[:, 4:5], x[:, 5:6],
                             x[:, 6:7], x[:, 7:8], x[:, 8:9], x[:, 9:10], x[:, 10:11]], dim=-1)
        scalar_proj = self.scalar_proj(scalars)
        ndv_feats = x[:, 13:16]
        return torch.cat([cat_emb, scalar_proj, ndv_feats], dim=-1)

    model._encode_nodes = patched_encode.__get__(model, PlanGNN)
    old_enc = model.node_encoder
    new_first = torch.nn.Linear(91, 128)
    new_first.weight.data[:, :88] = old_enc[0].weight.data
    new_first.bias.data = old_enc[0].bias.data
    old_enc[0] = new_first
    model.node_encoder = old_enc

    # Classification heads
    model.cpu_cls_head = torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                              torch.nn.Dropout(0.1), torch.nn.Linear(64, N_CLASSES))
    model.mem_cls_head = torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                              torch.nn.Dropout(0.1), torch.nn.Linear(64, N_CLASSES))

    def fwd(self, data):
        x = self._encode_nodes(data.x); x = self.node_encoder(x)
        e = self._encode_edges(data.edge_attr)
        for c, n in zip(self.convs, self.norms): x = n(x + c(x, data.edge_index, edge_attr=e))
        from torch_geometric.nn import global_add_pool, global_max_pool
        hm = global_max_pool(x, data.batch)
        gl = self.gate_mlp(x).squeeze(-1)
        hg = torch.stack([(torch.softmax(gl[data.batch == g], dim=0).unsqueeze(0) @
                           x[data.batch == g]).squeeze(0) for g in range(int(data.batch.max()) + 1)])
        hs = global_add_pool(x, data.batch)
        plan_emb = self.out_proj(hm + hg + hs)
        ns2 = torch.cat([data.x[:, 1:2], data.x[:, 3:4], data.x[:, 4:5], data.x[:, 5:6],
                         data.x[:, 6:7], data.x[:, 7:8], data.x[:, 8:9], data.x[:, 9:10],
                         data.x[:, 10:11]], -1)
        gs = global_add_pool(ns2, data.batch)
        nn2 = torch.bincount(data.batch + 1)[1:].float().unsqueeze(1)
        gf = self.global_skip(torch.cat([gs, nn2], -1))
        pa = torch.cat([plan_emb, gf], -1)
        return {"disk": self.disk_head(pa), "cpu_reg": self.cpu_head(pa),
                "cpu_cls": self.cpu_cls_head(pa), "mem_cls": self.mem_cls_head(pa)}

    model.forward = fwd.__get__(model, PlanGNN)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ─── Train ───
    tl = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    vl = DataLoader([graphs[i] for i in val_idx], batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 400), eta_min=1e-6)
    bv, bs = float('inf'), None

    for ep in range(1, args.epochs + 1):
        model.train()
        for data in tl:
            opt.zero_grad(); p = model(data)
            rl = F.huber_loss(p['disk'].squeeze(-1), data.y_disk.squeeze(-1)) \
                 + F.huber_loss(p['cpu_reg'].squeeze(-1), data.y_cpu_reg.squeeze(-1))
            cl = F.cross_entropy(p['cpu_cls'], data.y_cpu_cls) + F.cross_entropy(p['mem_cls'], data.y_mem_cls)
            (rl + cl).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if ep % 50 == 0:
            model.eval(); vl2, nb = 0.0, 0
            with torch.no_grad():
                for data in vl:
                    p = model(data)
                    vl2 += (F.huber_loss(p['disk'].squeeze(-1), data.y_disk.squeeze(-1))
                            + F.huber_loss(p['cpu_reg'].squeeze(-1), data.y_cpu_reg.squeeze(-1))
                            + F.cross_entropy(p['cpu_cls'], data.y_cpu_cls)
                            + F.cross_entropy(p['mem_cls'], data.y_mem_cls)).item(); nb += 1
            vl2 /= max(nb, 1)
            if vl2 < bv: bv = vl2; bs = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'E{ep:3d} val={vl2:.4f} best={bv:.4f}')

    model.load_state_dict(bs); model.eval()
    os.makedirs(f'{BASE}/checkpoints', exist_ok=True)

    # Save checkpoint with convergence check
    ckpt_path = f'{BASE}/checkpoints/best_final.pt'
    torch.save(bs, ckpt_path)

    # Check if training actually converged
    initial_loss = 2.35  # approximate baseline loss at init
    improvement = (initial_loss - bv) / initial_loss * 100
    print(f'\n  Val loss: {initial_loss:.4f} → {bv:.4f} (improvement: {improvement:.1f}%)')

    if improvement < 10:
        print(f'  WARNING: Model barely improved. Training may have failed.')
        print(f'  Re-run with --seed N to try a different initialization.')
    else:
        print(f'  Saved to {ckpt_path}')

    # ─── Test ───
    td = next(iter(DataLoader([graphs[i] for i in test_idx], batch_size=len(test_idx))))
    with torch.no_grad(): preds = model(td)

    # Disk regression (log-space Q-error)
    p = preds['disk'].squeeze(-1).numpy(); t = td.y_disk.squeeze(-1).numpy()
    pl = p * disk_std + disk_mean; tl = t * disk_std + disk_mean
    dqe = np.exp(np.abs(pl - tl)); dqs = np.sort(dqe); nq = len(dqs)
    d_r2 = 1 - np.sum((tl - pl) ** 2) / np.sum((tl - np.mean(tl)) ** 2)

    # CPU regression (log-space Q-error)
    p = preds['cpu_reg'].squeeze(-1).numpy(); t = td.y_cpu_reg.squeeze(-1).numpy()
    cpl = p * cpu_std + cpu_mean; ctl = t * cpu_std + cpu_mean
    cqe = np.exp(np.abs(cpl - ctl)); cqs = np.sort(cqe); ncq = len(cqs)
    c_r2 = 1 - np.sum((ctl - cpl) ** 2) / np.sum((ctl - np.mean(ctl)) ** 2)

    # Network LR
    npred_log = net_lr.predict(exch_log[test_idx].reshape(-1, 1))
    npred_r = np.maximum(np.exp(npred_log) - 1, 0)
    ntrue_r = np.exp(net_log[test_idx]) - 1
    nqe = np.maximum(npred_r / np.maximum(ntrue_r, 1), np.maximum(ntrue_r, 1) / np.maximum(npred_r, 1))
    nqs = np.sort(nqe); nq2 = len(nqs)

    # Classification
    nm = ['LOW', 'MED', 'HIGH']

    print(f"\n{'='*65}")
    print(f"FINAL RESULTS (test={len(test_idx)} queries, train={len(train_idx)})")
    print(f"{'='*65}")
    print(f"{'Disk  IO':>10s}  GNN reg  P50={dqs[nq // 2]:.2f}  P90={dqs[int(nq * .9)]:.2f}  "
          f"P95={dqs[int(nq * .95)]:.2f}  R²={d_r2:.4f}")
    print(f"{'CPU Res':>10s}  GNN reg  P50={cqs[ncq // 2]:.2f}  P90={cqs[int(ncq * .9)]:.2f}  "
          f"P95={cqs[int(ncq * .95)]:.2f}  R²={c_r2:.4f}")
    print(f"{'Net   IO':>10s}  estrs LR P50={nqs[nq2 // 2]:.2f}  P90={nqs[int(nq2 * .9)]:.2f}  "
          f"P95={nqs[int(nq2 * .95)]:.2f}")

    for k, lb in [('cpu', 'CPU-Res'), ('mem', 'Memory')]:
        p = preds[f'{k}_cls'].argmax(dim=1).numpy()
        t = td.__getattr__(f'y_{k}_cls').numpy()
        acc = np.mean(p == t)
        bacc = np.mean([np.mean(p[t == c] == c) for c in range(N_CLASSES)])
        bl = np.mean(t == 1)
        pc = ' | '.join([f'{nm[c]}:{np.mean(p[t == c] == c) * 100:.0f}%' for c in range(N_CLASSES)])
        print(f'{lb:>10s}  GNN cls  Acc={acc:.4f}  BalAcc={bacc:.4f}  base={bl:.4f}  {pc}')


if __name__ == '__main__':
    main()
