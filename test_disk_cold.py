"""Test disk IO with cold cache (after dropping page cache)."""
import subprocess, time, re, os

qid = '6409'
with open(f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql') as f:
    sql = f.read().strip().rstrip(';')

# Build io.stat paths
paths = {}
for name in ['tidb4', 'tidb5']:
    cid = subprocess.run(['docker', 'inspect', name, '--format', '{{.Id}}'],
                         capture_output=True, text=True).stdout.strip()
    paths[name] = f'/sys/fs/cgroup/system.slice/docker-{cid}.scope/io.stat'

def read_io():
    result = {}
    for name, path in paths.items():
        with open(path) as f: line = f.read().strip()
        m = re.search(r'rbytes=(\d+)', line); rb = int(m.group(1)) if m else 0
        m = re.search(r'wbytes=(\d+)', line); wb = int(m.group(1)) if m else 0
        result[name] = rb + wb
    return result

# Drop page caches first
print("Dropping page caches in TiFlash containers...")
for name in ['tidb4', 'tidb5']:
    subprocess.run(['docker', 'exec', name, 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])
    print(f"  {name}: cache dropped")
time.sleep(2)

# Run with cold cache
before = read_io()
print(f"\nExecuting query {qid} (cold cache)...")
start = time.time()
subprocess.run(["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root",
                "-D", "tpch_sf40", "-e", sql], capture_output=True, text=True, timeout=180)
elapsed = time.time() - start
after = read_io()

print(f"\nDone: {elapsed:.1f}s")
for name in paths:
    delta = after[name] - before[name]
    print(f"  {name}: ΔIO={delta/1e9:.2f} GB")

total = sum(after[n] - before[n] for n in paths)
print(f"\n  Total cold IO: {total/1e9:.2f} GB")
print(f"  Formula data_scanned_rows: 698M rows")
print(f"  Estimated bytes (×100B/row): 69.8 GB")
print(f"  Ratio: {69.8 / (total/1e9):.1f}x")
