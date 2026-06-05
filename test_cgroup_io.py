"""Compare cgroup block IO vs EXPLAIN ANALYZE disk formula."""
import subprocess, time, os, sys, re, threading
from collections import defaultdict

# Build cgroup paths
paths = {}
for name in ['tidb1','tidb2','tidb3','tidb4','tidb5']:
    cid = subprocess.run(['docker','inspect',name,'--format','{{.Id}}'],
                         capture_output=True, text=True).stdout.strip()
    p = f'/sys/fs/cgroup/system.slice/docker-{cid}.scope/io.stat'
    if os.path.exists(p): paths[name] = p

qid = '6409'
with open(f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql') as f:
    sql = f.read().strip().rstrip(';')

def read_io():
    """Read rbytes + wbytes for all containers."""
    result = {}
    for name, path in paths.items():
        try:
            with open(path) as f:
                line = f.read().strip()
            m = re.search(r'rbytes=(\d+)', line)
            rb = int(m.group(1)) if m else 0
            m = re.search(r'wbytes=(\d+)', line)
            wb = int(m.group(1)) if m else 0
            result[name] = {'read': rb, 'write': wb, 'total': rb + wb}
        except: pass
    return result

print(f"Query {qid}: 698M rows scanned (formula estimate)")

# Baseline
before = read_io()
print("\nBaseline cgroup IO:")
for name, io in before.items():
    if io: print(f"  {name}: read={io['read']/1e9:.2f}GB write={io['write']/1e9:.2f}GB")

# Sampling during execution at 1Hz
samples = []
start = time.time()
stop = False

def sampler():
    while not stop:
        t = time.time() - start
        io = read_io()
        for name, io_data in io.items():
            samples.append((t, name, io_data['read'], io_data['write']))
        time.sleep(1.0)

t = threading.Thread(target=sampler, daemon=True)
t.start()

print(f"\nExecuting...")
result = subprocess.run(
    ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-D", "tpch_sf40", "-e", sql],
    capture_output=True, text=True, timeout=120)
elapsed = time.time() - start
stop = True
time.sleep(0.5)

print(f"Done: {elapsed:.1f}s | {len(samples)} samples\n")

after = read_io()

# Compute deltas per container
print(f"{'Container':<10s} {'ΔRead':>12s} {'ΔWrite':>12s} {'ΔTotal':>12s}")
total_delta = 0
for name in sorted(paths.keys()):
    dr = after[name]['read'] - before[name]['read'] if name in after and name in before else 0
    dw = after[name]['write'] - before[name]['write'] if name in after and name in before else 0
    dt = dr + dw
    total_delta += dt
    fmt = lambda b: f"{b/1e9:.2f}GB" if b>1e9 else f"{b/1e6:.1f}MB" if b>1e6 else f"{b:.0f}B"
    print(f"{name:<10s} {fmt(dr):>12s} {fmt(dw):>12s} {fmt(dt):>12s}")

# Formula
sys.path.insert(0, '/home/anqian/Desktop/my_lab/workloads/gnn')
from train_ndv import parse_analyze, load_ndv_cache
ndv = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
with open(f'/home/anqian/Desktop/my_lab/workloads/explain_analyze_results/{qid}.txt') as f:
    lab = parse_analyze(f.read(), ndv)

# Our disk formula: data_scanned_rows = row count, need to convert to bytes
# Each scan reads specific columns, so bytes ≈ rows × avg_columns_width
# But for TPC-H, let's use a rough estimate: ~100 bytes per row on average
disk_rows = lab['disk_io_rows']
est_bytes = disk_rows * 100  # rough row width estimate

print(f"\n  EXPLAIN ANALYZE disk_io_rows: {disk_rows/1e6:.0f}M rows")
print(f"  Estimated bytes (×100B/row):  {est_bytes/1e9:.2f} GB")
print(f"  Cgroup IO delta (total):       {total_delta/1e9:.2f} GB")
print(f"  Ratio (formula / cgroup):      {est_bytes/max(total_delta,1):.2f}x")
