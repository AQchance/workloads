"""Test Docker memory monitoring during query execution."""
import subprocess, time, re, threading
from collections import defaultdict

def docker_stats():
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
        capture_output=True, text=True)
    result = {}
    for line in out.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) == 2 and 'tidb' in parts[0]:
            m = re.search(r'([\d.]+)(GiB|MiB|KiB|B)\s*/\s*([\d.]+)(GiB|MiB|KiB|B)', parts[1])
            if m:
                mult = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3}
                result[parts[0]] = float(m.group(1)) * mult[m.group(2)]
    return result

qid = '25555'
with open(f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql') as f:
    sql = f.read().strip().rstrip(';')

print(f"Query {qid}: {len(sql)} chars\n")
print("Baseline Docker memory:")
bl = docker_stats()
for n, v in bl.items():
    print(f"  {n}: {v/1e9:.2f} GB")

# Sampling thread
samples = []
start = time.time()
stop = False

def sampler():
    while not stop:
        t = time.time() - start
        mems = docker_stats()
        for n, v in mems.items():
            samples.append((t, n, v))
        time.sleep(0.3)

t = threading.Thread(target=sampler, daemon=True)
t.start()

# Execute
print(f"\nExecuting...")
result = subprocess.run(
    ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-D", "tpch_sf40", "-e", sql],
    capture_output=True, text=True, timeout=120)
elapsed = time.time() - start
stop = True
time.sleep(0.5)

print(f"Done: {elapsed:.1f}s\n")

# Per-container peaks
peaks = defaultdict(float)
for ts, name, mem in samples:
    peaks[name] = max(peaks[name], mem)

print(f"{'Container':<10s} {'Baseline':>12s} {'Peak':>12s} {'Delta':>12s}")
total = 0
for name in sorted(bl.keys()):
    bl_mem = bl.get(name, 0)
    pk = peaks.get(name, 0)
    dt = max(0, pk - bl_mem)
    total += dt
    fmt = lambda b: f"{b/1e9:.2f}GB" if b>1e9 else f"{b/1e6:.1f}MB"
    print(f"{name:<10s} {fmt(bl_mem):>12s} {fmt(pk):>12s} {fmt(dt):>12s}")

print(f"\n  Total Docker peak delta: {total/1e9:.2f} GB ({total/1e6:.0f} MB)")

# Get formula estimate
import sys
sys.path.insert(0, '/home/anqian/Desktop/my_lab/workloads/gnn')
from train_ndv import parse_analyze, load_ndv_cache
ndv = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
with open(f'/home/anqian/Desktop/my_lab/workloads/explain_analyze_results/{qid}.txt') as f:
    lab = parse_analyze(f.read(), ndv)
print(f"  Our formula estimate:    {lab['memory_bytes']/1e9:.2f} GB")
print(f"  Ratio (formula/Docker):  {lab['memory_bytes']/max(total,1):.2f}x")
