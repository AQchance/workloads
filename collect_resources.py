#!/usr/bin/env python3
"""
Collect per-container physical resource metrics for all queries.

Measures:
  - Memory:   cgroup memory.current at 5Hz, records peak per container
  - Network:  Docker NET I/O delta per container (before/after)
  - Disk IO:  cgroup io.stat delta per container (before/after, cold cache)
  - Latency:  wall-clock execution time

Hard timeout: 10 minutes per query. Timeout queries are skipped.
Resume: skips queries already in output directory.
"""

import subprocess, time, os, json, re, threading, signal
from collections import defaultdict

# ─── Configuration ───
ANALYZE_DIR = "/home/anqian/Desktop/my_lab/workloads/explain_analyze_results"
SQL_DIR = "/home/anqian/Desktop/my_lab/workloads/SQLStorm"
OUT_DIR = "/home/anqian/Desktop/my_lab/workloads/cgroup_resources"
MYSQL_CMD = "mysql -h 172.19.0.11 -P 4000 -u root -D tpch_sf40"

CONTAINERS = ["tidb1", "tidb2", "tidb3", "tidb4", "tidb5"]
TIFLASH = ["tidb4", "tidb5"]
COOLDOWN_S = 10
MEMORY_HZ = 5
HARD_TIMEOUT_S = 600  # 10 minutes per query
SUBPROCESS_TIMEOUT = 15  # timeout for docker/shell subprocess calls
RESTART_WAIT_S = 30  # wait after TiDB restart before continuing
MAX_CONSECUTIVE_FAILS = 5  # restart TiDB after this many consecutive failures

# ─── Cgroup paths ───
CGROUP_PATHS = {}
for name in CONTAINERS:
    cid = subprocess.run(
        ["docker", "inspect", name, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT
    ).stdout.strip()
    mem_path = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current"
    io_path = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/io.stat"
    if os.path.exists(mem_path):
        CGROUP_PATHS[name] = {"memory": mem_path, "io": io_path}


def _safe_subprocess(args, timeout=SUBPROCESS_TIMEOUT):
    """Run subprocess with timeout, return (ok, stdout)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def read_memory():
    result = {}
    for name, paths in CGROUP_PATHS.items():
        try:
            with open(paths["memory"]) as f:
                result[name] = int(f.read().strip())
        except:
            result[name] = 0
    return result


def read_disk_io():
    result = {}
    for name, paths in CGROUP_PATHS.items():
        try:
            with open(paths["io"]) as f:
                line = f.read().strip()
            m = re.search(r'rbytes=(\d+)', line); rb = int(m.group(1)) if m else 0
            m = re.search(r'wbytes=(\d+)', line); wb = int(m.group(1)) if m else 0
            result[name] = rb + wb
        except:
            result[name] = 0
    return result


def read_network():
    """Read precise network byte counters from /sys/class/net/eth0/statistics.
    Much more accurate than Docker stats (which rounds to GB)."""
    result = {}
    for name in CONTAINERS:
        try:
            rx = subprocess.run(
                ["docker", "exec", name, "cat", "/sys/class/net/eth0/statistics/rx_bytes"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            tx = subprocess.run(
                ["docker", "exec", name, "cat", "/sys/class/net/eth0/statistics/tx_bytes"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            result[name] = int(rx) + int(tx)
        except:
            result[name] = 0
    return result


def drop_caches():
    for name in TIFLASH:
        _safe_subprocess(
            ["docker", "exec", name, "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            timeout=10)


def execute_query_with_timeout(sql):
    """Execute query via mysql CLI with hard 10-minute timeout."""
    try:
        r = subprocess.run(
            ["timeout", str(HARD_TIMEOUT_S), "sh", "-c",
             f'{MYSQL_CMD} -e "{sql}"'],
            capture_output=True, text=True,
            timeout=HARD_TIMEOUT_S + 30)
        if r.returncode == 124:  # timeout command's exit code
            return False, 0, "TIMEOUT"
        return r.returncode == 0, 0, r.stderr  # we don't track actual time this way
    except subprocess.TimeoutExpired:
        return False, 0, "TIMEOUT"


def check_tidb_alive():
    """Quick check if TiDB is responding."""
    try:
        r = subprocess.run(
            ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-e", "SELECT 1"],
            capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "1" in r.stdout
    except:
        return False


def restart_tidb():
    """Restart TiDB cluster using the start script in tidb1."""
    print("\n*** TiDB appears DOWN, restarting... ***")
    try:
        subprocess.run(
            ["docker", "exec", "tidb1", "bash", "/root/tidb_start.sh"],
            capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("  WARNING: restart script timed out, waiting anyway...")
    # Wait for TiDB to be ready
    for attempt in range(30):
        if check_tidb_alive():
            print(f"*** TiDB restarted and responding (attempt {attempt+1}) ***\n")
            time.sleep(5)  # extra settle time
            return True
        time.sleep(2)
    print("*** FAILED to restart TiDB after 60s! ***\n")
    return False


def collect_one_query(qid):
    """Execute one query and collect all resource metrics. Returns dict or None."""
    sql_file = os.path.join(SQL_DIR, f"{qid}.sql")
    if not os.path.exists(sql_file):
        return None
    with open(sql_file) as f:
        sql = f.read().strip().rstrip(";")
    if not sql:
        return None
    # Escape quotes for shell
    sql_escaped = sql.replace("'", "'\\''")

    # ─── Preparation ───
    time.sleep(COOLDOWN_S)
    drop_caches()
    time.sleep(1)

    # ─── Baselines ───
    base_net = read_network()
    base_disk = read_disk_io()

    # ─── Memory sampling (5Hz) ───
    mem_samples = []
    stop_sampler = threading.Event()
    samp_started = threading.Event()

    def sampler():
        samp_started.set()
        while not stop_sampler.is_set():
            t = time.time()
            mems = read_memory()
            mem_samples.append((t, mems))
            elapsed = time.time() - t
            sleep_time = 1.0 / MEMORY_HZ - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    samp_thread = threading.Thread(target=sampler, daemon=True)
    samp_thread.start()
    samp_started.wait(timeout=2)

    # ─── Execute query ───
    start_time = time.time()
    timed_out = False
    exit_code = -1
    try:
        r = subprocess.run(
            ["timeout", str(HARD_TIMEOUT_S)] + MYSQL_CMD.split() + ["-e", sql],
            capture_output=True, text=True, timeout=HARD_TIMEOUT_S + 30)
        elapsed = time.time() - start_time
        exit_code = r.returncode
        timed_out = (exit_code == 124)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        timed_out = True

    stop_sampler.set()
    samp_thread.join(timeout=3)

    if timed_out:
        return {"qid": qid, "status": "timeout", "elapsed_s": round(elapsed, 1)}

    if exit_code != 0:
        return None  # query failed

    # ─── After execution ───
    after_net = read_network()
    after_disk = read_disk_io()

    # ─── Compute metrics ───
    latency_s = round(elapsed, 3)

    net_delta = {}
    for name in CONTAINERS:
        net_delta[name] = max(0, after_net.get(name, 0) - base_net.get(name, 0))

    disk_delta = {}
    for name in CONTAINERS:
        disk_delta[name] = max(0, after_disk.get(name, 0) - base_disk.get(name, 0))

    mem_peak = defaultdict(int)
    mem_bl = None
    for ts, mems in mem_samples:
        if mem_bl is None:
            mem_bl = dict(mems)
        for name in CONTAINERS:
            if name in mems:
                mem_peak[name] = max(mem_peak[name], mems[name])

    mem_delta = {}
    if mem_bl:
        for name in CONTAINERS:
            mem_delta[name] = max(0, mem_peak.get(name, 0) - mem_bl.get(name, 0))

    return {
        "qid": qid,
        "status": "ok",
        "latency_s": latency_s,
        "memory_peak_bytes": {n: mem_peak.get(n, 0) for n in CONTAINERS},
        "memory_delta_bytes": mem_delta,
        "network_delta_bytes": net_delta,
        "disk_delta_bytes": disk_delta,
        "n_memory_samples": len(mem_samples),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_queries = sorted([
        f.replace(".sql", "") for f in os.listdir(SQL_DIR)
        if f.endswith(".sql") and f.replace(".sql", "").isdigit()
    ], key=int)

    done = set()
    for f in os.listdir(OUT_DIR):
        if f.endswith(".json"):
            done.add(f.replace(".json", ""))

    pending = [q for q in all_queries if q not in done]
    total = len(all_queries)
    remaining = len(pending)

    print(f"Resource collection")
    print(f"  Total: {total} | Done: {len(done)} | Pending: {remaining}")
    print(f"  Timeout: {HARD_TIMEOUT_S}s per query")
    print(f"  Output: {OUT_DIR}\n")

    n_ok = 0
    n_fail = 0
    n_timeout = 0
    consecutive_fails = 0

    for i, qid in enumerate(pending):
        # Check TiDB health and auto-restart if needed
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            if not check_tidb_alive():
                restart_tidb()
                consecutive_fails = 0
                time.sleep(RESTART_WAIT_S)

        print(f"[{i+1}/{remaining}] Q{qid} ...", end=" ", flush=True)
        data = collect_one_query(qid)

        if data is None:
            print("FAIL")
            n_fail += 1
            consecutive_fails += 1
            continue

        consecutive_fails = 0  # reset on success or timeout

        if data.get("status") == "timeout":
            print(f"TIMEOUT ({data['elapsed_s']:.0f}s)")
            # Save timeout record
            with open(os.path.join(OUT_DIR, f"{qid}.json"), "w") as f:
                json.dump(data, f)
            n_timeout += 1
            consecutive_fails += 1  # timeout might mean DB struggling
            continue

        with open(os.path.join(OUT_DIR, f"{qid}.json"), "w") as f:
            json.dump(data, f)

        net_t = sum(data["network_delta_bytes"].values())
        mem_t = sum(data["memory_delta_bytes"].values())
        disk_t = sum(data["disk_delta_bytes"].values())

        def fmt(b):
            if b > 1e9: return f"{b/1e9:.1f}G"
            if b > 1e6: return f"{b/1e6:.1f}M"
            return f"{b/1e3:.1f}K"
        print(f"OK {data['latency_s']:.0f}s m={fmt(mem_t)} n={fmt(net_t)} d={fmt(disk_t)}")
        n_ok += 1

        if (n_ok + n_fail + n_timeout) % 25 == 0:
            print(f"  --- progress: {n_ok} ok, {n_fail} fail, {n_timeout} timeout ---")

    # Summary
    print(f"\nDone! OK={n_ok} FAIL={n_fail} TIMEOUT={n_timeout}")

    # Generate summary CSV
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.endswith(".json"): continue
        with open(os.path.join(OUT_DIR, f)) as fp:
            d = json.load(fp)
        qid = d["qid"]
        lat = d.get("latency_s", d.get("elapsed_s", 0))
        status = d.get("status", "ok")
        mem = sum(d.get("memory_delta_bytes", {}).values()) if status == "ok" else -1
        net = sum(d.get("network_delta_bytes", {}).values()) if status == "ok" else -1
        disk = sum(d.get("disk_delta_bytes", {}).values()) if status == "ok" else -1
        rows.append(f"{qid},{status},{lat},{mem},{net},{disk}")

    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write("qid,status,latency_s,memory_delta_bytes,network_delta_bytes,disk_delta_bytes\n")
        f.write("\n".join(sorted(rows)) + "\n")


if __name__ == "__main__":
    main()
