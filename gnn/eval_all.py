"""Comprehensive evaluation of all 4 resource dimensions."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ndv import load_dataset, load_ndv_cache, load_dist_cache, normalize_labels
from model import PlanGNN
from torch_geometric.data import Batch

ndv = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
dist = load_dist_cache('/home/anqian/Desktop/my_lab/workloads/dist_cache.json')
graphs, labels, meta = load_dataset(
    '/home/anqian/Desktop/my_lab/workloads/explain_plans',
    '/home/anqian/Desktop/my_lab/workloads/explain_analyze_results', ndv, dist)
norm_labels, stats = normalize_labels(labels)
key_map = {'memory_bytes': 'mem', 'disk_io_rows': 'disk', 'network_bytes': 'net', 'latency_ms': 'cpu'}
for g, nl in zip(graphs, norm_labels):
    for rk, sk in key_map.items():
        setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

torch.manual_seed(42); np.random.seed(42)
n = len(graphs)
indices = np.random.permutation(n)
test_idx = indices[int(n * 0.85):]

ckpt = torch.load('/home/anqian/Desktop/my_lab/workloads/checkpoints/best_dist.pt',
                  weights_only=True, map_location='cpu')
model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
model.load_state_dict(ckpt)
model.eval()

rmap = {'mem': 'memory_bytes', 'disk': 'disk_io_rows', 'net': 'network_bytes', 'cpu': 'latency_ms'}
dim_labels = {'mem': 'Memory (bytes)', 'disk': 'Disk IO (rows)', 'net': 'Network (bytes)', 'cpu': 'Latency (ms)'}

test_preds = {k: [] for k in key_map.values()}
test_targets = {k: [] for k in key_map.values()}

with torch.no_grad():
    for i in test_idx:
        g = graphs[i]
        out = model(Batch.from_data_list([g]))
        for k in key_map.values():
            test_preds[k].append(out[k].item())
            test_targets[k].append(g.y_mem.item() if k=='mem' else
                                 (g.y_disk.item() if k=='disk' else
                                  (g.y_net.item() if k=='net' else g.y_cpu.item())))

print(f"Dataset: {len(graphs)} queries | Test: {len(test_idx)} queries | Model: PlanGNN 347K params\n")
print(f"{'Dimension':<16s} {'P50':>8s} {'P80':>8s} {'P90':>8s} {'P95':>8s} {'P99':>8s} {'R²':>8s} {'<2x':>8s} {'<5x':>8s}")
print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for k, label in [('mem','Memory'), ('disk','DiskIO'), ('net','Network'), ('cpu','Latency')]:
    p = np.array(test_preds[k])
    t = np.array(test_targets[k])
    std_k = stats[rmap[k]]['std']
    mn_k = stats[rmap[k]]['mean']

    p_raw = np.maximum(np.exp(p * std_k + mn_k) - 1, 0)
    t_raw = np.maximum(np.exp(t * std_k + mn_k) - 1, 0)

    qe = np.maximum(p_raw / np.maximum(t_raw, 1), np.maximum(t_raw, 1) / np.maximum(p_raw, 1))
    qs = np.sort(qe)
    nq = len(qs)

    ss_res = np.sum((p - t)**2)
    ss_tot = np.sum((t - np.mean(t))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    pcts = [50, 80, 90, 95, 99]
    vals = [f"{qs[int(nq*pct/100)]:.2f}" for pct in pcts]

    within_2x = f"{np.mean(qe <= 2.0)*100:.0f}%"
    within_5x = f"{np.mean(qe <= 5.0)*100:.0f}%"

    print(f"{label:<16s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s} {vals[4]:>8s} {r2:>7.4f} {within_2x:>8s} {within_5x:>8s}")

print(f"\nMemory label: actRows-based formula with estRows fallback")
print(f"DiskIO label: SUM(data_scanned_rows from EXPLAIN ANALYZE)")
print(f"Network label: SUM(Exchange actRows × column avg_width)")
print(f"Latency label: root operator wall-clock time from EXPLAIN ANALYZE")
