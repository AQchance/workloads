"""Run multiple high-network queries and compare formula vs Docker NET I/O."""
import subprocess, time, re, os, sys, statistics

sys.path.insert(0, '/home/anqian/Desktop/my_lab/workloads/gnn')
from train_ndv import load_ndv_cache
ndv = load_ndv_cache('/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')

def read_docker_net():
    """Read cumulative NET I/O for all tidb containers (bytes)."""
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.NetIO}}"],
        capture_output=True, text=True).stdout
    result = {}
    for line in out.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) != 2 or 'tidb' not in parts[0]: continue
        name = parts[0]
        m = re.search(r'([\d.]+)(\w+)\s*/\s*([\d.]+)(\w+)', parts[1])
        if m:
            mult = {'B':1, 'KB':1024, 'MB':1024**2, 'GB':1024**3, 'TB':1024**4}
            net_in = float(m.group(1)) * mult[m.group(2)]
            net_out = float(m.group(3)) * mult[m.group(4)]
            result[name] = net_in + net_out
    return result

def run_one(qid):
    sql_file = f'/home/anqian/Desktop/my_lab/workloads/SQLStorm/{qid}.sql'
    if not os.path.exists(sql_file): return None
    with open(sql_file) as f: sql = f.read().strip().rstrip(';')

    from train_ndv import parse_analyze
    analyze_file = f'/home/anqian/Desktop/my_lab/workloads/explain_analyze_results/{qid}.txt'
    if not os.path.exists(analyze_file): return None
    with open(analyze_file) as f: lab = parse_analyze(f.read(), ndv)

    before = read_docker_net()
    start = time.time()
    subprocess.run(["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root",
                    "-D", "tpch_sf40", "-e", sql], capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start
    after = read_docker_net()

    # TiFlash nodes only
    actual = sum(after[n] - before[n] for n in ['tidb4', 'tidb5'] if n in after)
    formula = lab['network_bytes']
    return formula, actual, elapsed, lab['latency_ms'] / 1000

# Queries to test - high network, reasonable runtime
queries = ['21019', '25436', '25411', '8060', '8462', '408', '8146']

print(f"{'QID':>6s} {'Formula':>12s} {'ActualIO':>12s} {'Ratio':>8s} {'Elapsed':>8s} {'Lat(ana)':>9s}")
print('-' * 62)

results = []
for qid in queries:
    r = run_one(qid)
    if r is None:
        print(f"{qid:>6s}  SKIP (file not found)")
        continue
    fm, act, ela, lat = r
    ratio = fm / max(act, 1)
    def fmt(b): return f"{b/1e9:.2f}GB" if b>1e9 else f"{b/1e6:.1f}MB"
    print(f"{qid:>6s} {fmt(fm):>12s} {fmt(act):>12s} {ratio:>7.2f}x {ela:>7.1f}s {lat:>8.1f}s")
    results.append((qid, fm, act, ratio, ela, lat))

if results:
    ratios = [r[3] for r in results]
    import statistics
    print(f"\nRatio stats: median={statistics.median(ratios):.1f}x "
          f"mean={statistics.mean(ratios):.1f}x "
          f"min={min(ratios):.1f}x max={max(ratios):.1f}x")
