"""Re-run query to test if block IO is from cache."""
import subprocess, time, re, os

qid = '6409'
with open(f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql') as f:
    sql = f.read().strip().rstrip(';')

paths = {}
for name in ['tidb4','tidb5']:
    cid = subprocess.run(['docker','inspect',name,'--format','{{.Id}}'],
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

for run_id in [2, 3]:
    before = read_io()
    print(f"Run {run_id}...")
    start = time.time()
    subprocess.run(["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root",
                    "-D", "tpch_sf40", "-e", sql], capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start
    after = read_io()
    print(f"  Done: {elapsed:.1f}s")
    for name in paths:
        delta = (after[name] - before[name]) / 1e6
        print(f"  {name}: delta={delta:.1f} MB")
