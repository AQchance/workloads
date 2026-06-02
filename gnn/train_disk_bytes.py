"""
Train PlanGNN with disk IO labels in BYTES (not rows).

Disk bytes = Σ (data_scanned_rows × table_row_width) per scan operator.
Table row widths are computed from NDV cache column avg_width values.

Same 4-regression architecture as train_ndv.py, only the disk label changes.
CPU (wall-clock), Memory, Network labels unchanged.
"""

import os, sys, re, math, json, argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN
from train_ndv import (OPERATOR_CLASS_MAP, LOCATION_MAP, JOIN_TYPE_MAP, EXCHANGE_TYPE_MAP)


def compute_table_row_widths(ndv_cache: dict) -> dict:
    """Sum avg_width of all columns per table. Returns {table_name: total_row_width_bytes}."""
    table_widths = defaultdict(float)
    for col, info in ndv_cache.items():
        table = col.split('.')[0]
        table_widths[table] += info.get('avg_width', 8)
    return dict(table_widths)


def extract_disk_bytes_from_analyze(analyze_text: str, table_widths: dict) -> float:
    """
    Compute total disk IO in bytes from EXPLAIN ANALYZE.
    disk_bytes = Σ (data_scanned_rows × table_row_width) per scan op.
    """
    lines = analyze_text.strip().split('\n')
    total_bytes = 0.0
    in_table = False

    for line in lines:
        if line.startswith('id\t'): in_table = True; continue
        if in_table and '\t' in line:
            parts = line.split('\t')
            if len(parts) < 6: continue
            raw_id = parts[0].strip()
            exec_info = parts[5].strip()
            access_obj = parts[4].strip() if len(parts) > 4 else ""

            op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
            op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
            op_name = re.sub(r'_\d+$', '', op_name)

            if op_name not in ('TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan'):
                continue

            # Get data_scanned_rows
            m = re.search(r'data_scanned_rows:(\d+)', exec_info)
            if not m:
                continue
            rows = float(m.group(1))

            # Get table name from access object or plan
            table = "unknown"
            if 'table:' in access_obj:
                table = re.search(r'table:(\w+)', access_obj).group(1)

            # Get row width for this table
            row_width = table_widths.get(table, 120)  # default ~120 bytes/row for TPC-H

            total_bytes += rows * row_width

    return total_bytes


def load_dataset_with_disk_bytes(plan_dir: str, analyze_dir: str, ndv_cache: dict):
    """
    Load plans + labels. Same as train_ndv but with disk IO in bytes.
    """
    table_widths = compute_table_row_widths(ndv_cache)
    print(f"  Table row widths: { {k: f'{v:.0f}B' for k, v in sorted(table_widths.items())} }")

    plan_files = set(f for f in os.listdir(plan_dir) if f.endswith('.txt'))
    analyze_files = set(f for f in os.listdir(analyze_dir) if f.endswith('.txt'))
    common = sorted(plan_files & analyze_files, key=lambda x: int(x.replace('.txt', '')))

    from train_ndv import parse_plan, parse_memory_bytes, parse_time_ms

    graphs, labels, meta = [], [], []
    for fname in common:
        with open(os.path.join(plan_dir, fname)) as f: plan_text = f.read()
        with open(os.path.join(analyze_dir, fname)) as f: analyze_text = f.read()

        plan_lines = [l for l in plan_text.strip().split('\n') if '\t' in l and not l.startswith('--')]
        if not plan_lines: continue
        g = parse_plan(plan_text, ndv_cache)
        if g is None: continue

        # Extract labels from ANALYZE
        lines = analyze_text.strip().split('\n')
        total_mem, total_net = 0.0, 0.0
        in_table = False
        for line in lines:
            if line.startswith('id\t'): in_table = True; continue
            if in_table and '\t' in line:
                parts = line.split('\t')
                if len(parts) < 8: continue
                raw_id = parts[0].strip()
                memory_str = parts[7].strip()
                act_rows_str = parts[2].strip()

                mem_val = parse_memory_bytes(memory_str)
                if mem_val: total_mem += mem_val

                on = re.sub(r'^[│├└─\s]+', '', raw_id)
                on = re.sub(r'\(Build\)|\(Probe\)', '', on).strip()
                on = re.sub(r'_\d+$', '', on)
                if on in ("ExchangeSender", "ExchangeReceiver"):
                    try: total_net += float(act_rows_str)
                    except: pass

        # CPU time (wall-clock) from root operator
        first_row = next((p for p in lines[lines.index(next(l for l in lines if l.startswith('id\t')))+1:]
                          if '\t' in p), '')
        root_exec = first_row.split('\t')[5] if '\t' in first_row else ""
        cpu_time = parse_time_ms(root_exec) if root_exec else 0

        # Disk IO in BYTES
        disk_bytes = extract_disk_bytes_from_analyze(analyze_text, table_widths)

        labels.append({
            "cpu_time_ms": max(cpu_time, 0),
            "memory_bytes": total_mem,
            "disk_io_bytes": disk_bytes,  # NEW: bytes instead of rows
            "network_rows": total_net,
        })
        graphs.append(g)
        meta.append(fname.replace('.txt', ''))

    return graphs, labels, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_plans')
    parser.add_argument('--analyze-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_analyze_results')
    parser.add_argument('--ndv-cache', default='/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ndv_cache = json.load(open(args.ndv_cache))

    print("Loading data with disk-byte labels...")
    graphs, labels_raw, meta = load_dataset_with_disk_bytes(args.plan_dir, args.analyze_dir, ndv_cache)
    print(f"  {len(graphs)} queries")

    # Custom normalize for disk_io_bytes key
    keys = ['cpu_time_ms', 'memory_bytes', 'disk_io_bytes', 'network_rows']
    log_labels = [{k: math.log(1.0 + max(l[k], 0)) for k in keys} for l in labels_raw]
    stats = {}
    for k in keys:
        vals = [l[k] for l in log_labels]
        stats[k] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-8)}
    norm_labels = [{k: (l[k] - stats[k]['mean']) / stats[k]['std'] for k in keys} for l in log_labels]
    gnn_keys = {'cpu_time_ms': 'cpu', 'memory_bytes': 'mem', 'disk_io_bytes': 'disk', 'network_rows': 'net'}
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in gnn_keys.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    n = len(graphs)
    np.random.seed(args.seed)
    idx = np.random.permutation(n)
    tr = idx[:int(n*.7)]; va = idx[int(n*.7):int(n*.85)]; te = idx[int(n*.85):]
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    # Show disk byte label distribution
    disk_raw = np.array([l['disk_io_bytes'] for l in labels_raw])
    disk_log = np.log(1 + disk_raw)
    print(f"  Disk bytes: median={np.median(disk_raw)/1e9:.2f}GB  P95={np.percentile(disk_raw,95)/1e9:.2f}GB  max={np.max(disk_raw)/1e9:.2f}GB")

    # Model (same monkey-patch as train_ndv.py)
    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
    orig_enc = model._encode_nodes

    def patched_encode(self, x):
        cat_emb = torch.cat([
            self.op_class_emb(x[:, 0].long()), self.location_emb(x[:, 2].long()),
            self.join_type_emb(x[:, 11].long()), self.exchange_type_emb(x[:, 12].long()),
        ], dim=-1)
        sp = self.scalar_proj(torch.cat([x[:, 1:2], x[:, 3:4], x[:, 4:5], x[:, 5:6],
                                         x[:, 6:7], x[:, 7:8], x[:, 8:9], x[:, 9:10],
                                         x[:, 10:11]], -1))
        return torch.cat([cat_emb, sp, x[:, 13:16]], -1)

    model._encode_nodes = patched_encode.__get__(model, PlanGNN)
    old_enc = model.node_encoder
    nf = torch.nn.Linear(91, 128)
    nf.weight.data[:, :88] = old_enc[0].weight.data
    nf.bias.data = old_enc[0].bias.data
    old_enc[0] = nf
    model.node_encoder = old_enc

    def fwd(self, data):
        x = self._encode_nodes(data.x); x = self.node_encoder(x)
        e = self._encode_edges(data.edge_attr)
        for c, n in zip(self.convs, self.norms): x = n(x + c(x, data.edge_index, edge_attr=e))
        from torch_geometric.nn import global_add_pool, global_max_pool
        hm = global_max_pool(x, data.batch)
        gl = self.gate_mlp(x).squeeze(-1)
        hg = torch.stack([(torch.softmax(gl[data.batch == g], dim=0).unsqueeze(0) @
                           x[data.batch == g]).squeeze(0) for g in range(int(data.batch.max()) + 1)])
        hs = global_add_pool(x, data.batch); pe = self.out_proj(hm + hg + hs)
        ns2 = torch.cat([data.x[:, 1:2], data.x[:, 3:4], data.x[:, 4:5], data.x[:, 5:6],
                         data.x[:, 6:7], data.x[:, 7:8], data.x[:, 8:9], data.x[:, 9:10],
                         data.x[:, 10:11]], -1)
        gf = self.global_skip(torch.cat([global_add_pool(ns2, data.batch),
                                         torch.bincount(data.batch + 1)[1:].float().unsqueeze(1)], -1))
        pa = torch.cat([pe, gf], -1)
        return {"cpu": self.cpu_head(pa), "mem": self.mem_head(pa),
                "disk": self.disk_head(pa), "net": self.net_head(pa)}

    model.forward = fwd.__get__(model, PlanGNN)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    tl = DataLoader([graphs[i] for i in tr], batch_size=args.batch_size, shuffle=True)
    vl = DataLoader([graphs[i] for i in va], batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 400), eta_min=1e-6)
    bv, bs = float('inf'), None

    for ep in range(1, args.epochs + 1):
        model.train()
        for data in tl:
            opt.zero_grad(); p = model(data)
            loss = sum(F.huber_loss(p[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                       for k in gnn_keys.values())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if ep % 50 == 0:
            model.eval(); vl2, nb = 0.0, 0
            with torch.no_grad():
                for data in vl:
                    p = model(data)
                    vl2 += sum(F.huber_loss(p[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                               for k in gnn_keys.values()).item(); nb += 1
            vl2 /= max(nb, 1)
            if vl2 < bv: bv = vl2; bs = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'E{ep:3d} val={vl2:.4f} best={bv:.4f}')

    model.load_state_dict(bs); model.eval()

    # Test evaluation
    td = next(iter(DataLoader([graphs[i] for i in te], batch_size=len(te))))
    with torch.no_grad(): preds = model(td)

    rmap = {'cpu': 'cpu_time_ms', 'mem': 'memory_bytes', 'disk': 'disk_io_bytes', 'net': 'network_rows'}
    print(f"\n{'='*65}")
    print(f"FINAL RESULTS (test={len(te)} queries) — Disk IO in BYTES")
    print(f"{'='*65}")

    for k, lb in [('disk', 'Disk IO (bytes)'), ('cpu', 'CPU (wall)'), ('mem', 'Memory'), ('net', 'Network')]:
        p = preds[k].squeeze(-1).numpy()
        t = td.__getattr__(f'y_{k}').squeeze(-1).numpy()
        sm = stats[rmap[k]]['std']; sn = stats[rmap[k]]['mean']
        pl = p * sm + sn; tl = t * sm + sn
        qe = np.exp(np.abs(pl - tl)); qs = np.sort(qe); nq = len(qs)
        r2 = 1 - np.sum((tl - pl) ** 2) / np.sum((tl - np.mean(tl)) ** 2)
        print(f'{lb:>18s}  P50={qs[nq // 2]:.2f}  P90={qs[int(nq * .9)]:.2f}  '
              f'P95={qs[int(nq * .95)]:.2f}  R²={r2:.4f}')


if __name__ == '__main__':
    main()
