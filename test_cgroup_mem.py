"""Test cgroup memory monitoring at 5Hz during query execution."""
import subprocess, time, re, threading, os
from collections import defaultdict

# Build cgroup paths
paths = {}
for name in ['tidb1','tidb2','tidb3','tidb4','tidb5']:
    cid = subprocess.run(['docker','inspect',name,'--format','{{.Id}}'],
                         capture_output=True, text=True).stdout.strip()
    p = f'/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current'
    if os.path.exists(p):
        paths[name] = p

qid = '25555'
with open(f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql') as f:
    sql = f.read().strip().rstrip(';')

print(f"Query {qid} | Sampling at 5Hz (200ms interval)")

# Baseline
bl = {}
for name, path in paths.items():
    with open(path) as f:
        bl[name] = int(f.read().strip())
    print(f"  {name} baseline: {bl[name]/1e9:.2f} GB")

# Sampling at 5Hz
samples = []
start = time.time()
stop = False

def sampler():
    while not stop:
        t = time.time() - start
        for name, path in paths.items():
            try:
                with open(path) as f:
                    mem = int(f.read().strip())
                samples.append((t, name, mem))
            except:
                pass
        time.sleep(0.2)

import threading
t = threading.Thread(target=sampler, daemon=True)
t.start()

# Execute
print(f"\nExecuting...")
result = subprocess.run(
    ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-D", "tpch_sf40", "-e", sql],
    capture_output=True, text=True, timeout=120)
elapsed = time.time() - start
stop = True
time.sleep(0.3)

print(f"Done: {elapsed:.1f}s | {len(samples)} samples collected\n")

# Per-container peaks
peaks = defaultdict(float)
for ts, name, mem in samples:
    peaks[name] = max(peaks[name], mem)

print(f"{'Container':<10s} {'Baseline':>12s} {'Peak':>12s} {'Delta':>12s}")
total = 0
for name in sorted(paths.keys()):
    bl_mem = bl.get(name, 0)
    pk = peaks.get(name, 0)
    dt = max(0, pk - bl_mem)
    total += dt
    fmt = lambda b: f"{b/1e9:.2f}GB" if b>1e9 else f"{b/1e6:.1f}MB"
    print(f"{name:<10s} {fmt(bl_mem):>12s} {fmt(pk):>12s} {fmt(dt):>12s}")

print(f"\n  Total Docker peak delta: {total/1e9:.2f} GB")

# Get formula estimate
import sys
sys.path.insert(0, '/home/anqian/Desktop/my_lab/workloads/gnn')
from train_ndv import parse_analyze, load_ndv_cache
ndv = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
with open(f'/home/anqian/Desktop/my_lab/workloads/explain_analyze_results/{qid}.txt') as f:
    lab = parse_analyze(f.read(), ndv)
print(f"  Our formula estimate:    {lab['memory_bytes']/1e9:.2f} GB")
print(f"  Ratio (formula/Docker):  {lab['memory_bytes']/max(total,1):.2f}x")

# Show TiFlash memory curve (first few and peaks)
print(f"\n  TiFlash memory trace (first 20 points + peak region):")
tiflash_samples = [(ts, name, mem) for ts, name, mem in samples if name in ('tidb4','tidb5')]
# Show every Nth sample to keep output reasonable
n_show = max(1, len(tiflash_samples) // 15)
for i, (ts, name, mem) in enumerate(tiflash_samples):
    delta = (mem - bl.get(name, 0)) / 1e9
    if i % n_show == 0 or delta > (peaks.get(name, 0) - bl.get(name, 0)) * 0.9 / 1e9:
        marker = " <-- PEAK" if mem >= peaks.get(name, 0) * 0.95 else ""
        print(f"    t={ts:5.1f}s  {name}: {mem/1e9:.2f}GB (Δ{delta:+.2f}GB){marker}")
