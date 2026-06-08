"""
Step 1: Extract GNN plan embeddings and resource predictions for all queries.

Runs trained PlanGNN over EXPLAIN plans, saves per-query:
  - plan_emb: 128-dim graph embedding
  - resources: 5 predicted scalars (mem, disk, net, lat, cpures)
  - serial_labels: cgroup-measured physical labels

Output: lstm/gnn_features.json
"""

import sys, os, json
import numpy as np
import torch
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gnn'))
from model import PlanGNN
from train_cgroup import load_cgroup_labels
from train_ndv import load_ndv_cache, load_dist_cache, parse_plan

TRACE_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv'
PLAN_DIR = '/home/anqian/Desktop/my_lab/workloads/explain_plans'
CGROUP_DIR = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
OUT_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'

# Load cgroup labels (serial runtimes + physical resources)
cgroup_labels = load_cgroup_labels(CGROUP_DIR)
print(f"Loaded {len(cgroup_labels)} cgroup labels")

# Find all unique query IDs in the concurrent trace
trace_qids = set()
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f):
        trace_qids.add(row['qid'])
print(f"Unique queries in trace: {len(trace_qids)}")

# Load NDV cache and dist cache for plan parsing
ndv_cache = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
dist_cache = load_dist_cache('/home/anqian/Desktop/my_lab/workloads/dist_cache.json')

# Load GNN model
model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
ckpt = torch.load('/home/anqian/Desktop/my_lab/workloads/checkpoints/best_cgroup.pt',
                  weights_only=True, map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Extract features for each query
features = {}
n_skipped = 0

for qid in sorted(trace_qids):
    pf = os.path.join(PLAN_DIR, f'{qid}.txt')
    if not os.path.exists(pf):
        n_skipped += 1
        continue

    with open(pf) as f:
        plan_text = f.read()

    g = parse_plan(plan_text, ndv_cache, dist_cache)
    if g is None or g.x.shape[0] == 0:
        n_skipped += 1
        continue

    # Forward pass
    with torch.no_grad():
        g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
        out = model(g)

    plan_emb = out['plan_emb'].squeeze(0).cpu().numpy().tolist()
    resources = {
        'mem': float(out['mem'].item()),
        'disk': float(out['disk'].item()),
        'net': float(out['net'].item()),
        'lat': float(out['lat'].item()),
        'cpures': float(out['cpures'].item()),
    }

    # Serial labels from cgroup
    serial = {}
    if qid in cgroup_labels:
        cl = cgroup_labels[qid]
        serial = {
            'memory_bytes': cl['memory_bytes'],
            'network_bytes': cl['network_bytes'],
            'disk_bytes': cl['disk_bytes'],
            'latency_s': cl['latency_s'],
            'cpu_resource_s': cl.get('cpu_resource_s', 0),
        }

    features[qid] = {
        'plan_emb': plan_emb,
        'gpu_resources': resources,  # GNN's own prediction (z-score space)
        'serial_labels': serial,
    }

print(f"Extracted features for {len(features)} queries, skipped {n_skipped}")

with open(OUT_FILE, 'w') as f:
    json.dump(features, f)

print(f"Saved to {OUT_FILE}")
print(f"File size: {os.path.getsize(OUT_FILE)/1024:.0f} KB")
