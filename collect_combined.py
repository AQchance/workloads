#!/usr/bin/env python3
"""
Combined EXPLAIN ANALYZE + cgroup resource collection.

For each query:
  1. Run EXPLAIN ANALYZE (TiDB executes the query for real)
  2. Simultaneously sample cgroup metrics at 5Hz
  3. Save EXPLAIN ANALYZE output → explain_analyze_results/{qid}.txt
  4. Save cgroup resource data       → cgroup_resources/{qid}.json

Auto-resume: skips queries that already have BOTH files.
TiDB crash: auto-restart, re-queue failed query.
"""

import subprocess, time, os, json, re, threading
from collections import defaultdict

SQL_DIR = "/home/anqian/Desktop/my_lab/workloads/SQLStorm"
PLAN_DIR = "/home/anqian/Desktop/my_lab/workloads/explain_plans"
ANALYZE_DIR = "/home/anqian/Desktop/my_lab/workloads/explain_analyze_results"
CGROUP_DIR = "/home/anqian/Desktop/my_lab/workloads/cgroup_resources"
RESULT_FILE = "/home/anqian/Desktop/my_lab/workloads/collect_combined_results.txt"

MYSQL_CMD = "mysql -h 172.19.0.11 -P 4000 -u root -D tpch_sf40"

CONTAINERS = ["tidb1", "tidb2", "tidb3", "tidb4", "tidb5"]
TIFLASH = ["tidb4", "tidb5"]
COOLDOWN_S = 10
MEMORY_HZ = 5
HARD_TIMEOUT_S = 600
SUBPROCESS_TIMEOUT = 15
RESTART_WAIT_S = 30
MAX_CONSECUTIVE_FAILS = 5

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
        subprocess.run(
            ["docker", "exec", name, "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=10)


def check_tidb_alive():
    try:
        r = subprocess.run(
            ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-e", "SELECT 1"],
            capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except:
        return False


def restart_tidb():
    print("\n*** TiDB appears DOWN, restarting... ***")
    try:
        subprocess.run(
            ["docker", "exec", "tidb1", "bash", "/root/tidb_start.sh"],
            capture_output=True, text=True, timeout=120)
    except:
        pass
    for attempt in range(30):
        if check_tidb_alive():
            print(f"*** TiDB restarted (attempt {attempt+1}) ***\n")
            time.sleep(5)
            return True
        time.sleep(2)
    print("*** FAILED to restart TiDB! ***\n")
    return False


def collect_one_query(qid):
    sql_file = os.path.join(SQL_DIR, f"{qid}.sql")
    if not os.path.exists(sql_file):
        return None, None
    with open(sql_file) as f:
        sql = f.read().strip().rstrip(";")
    if not sql:
        return None, None

    # ─── Preparation ───
    time.sleep(COOLDOWN_S)
    drop_caches()
    time.sleep(1)

    # ─── Baselines ───
    base_net = read_network()
    base_disk = read_disk_io()

    # ─── Memory sampling (5Hz background thread) ───
    mem_samples = []
    stop_sampler = threading.Event()

    def sampler():
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
    time.sleep(0.5)  # let sampler start

    # ─── Execute EXPLAIN ANALYZE ───
    start_time = time.time()
    timed_out = False
    exit_code = -1
    analyze_output = ""
    try:
        r = subprocess.run(
            ["timeout", str(HARD_TIMEOUT_S)] + MYSQL_CMD.split() + ["-e", f"EXPLAIN ANALYZE {sql}"],
            capture_output=True, text=True, timeout=HARD_TIMEOUT_S + 30)
        elapsed = time.time() - start_time
        exit_code = r.returncode
        timed_out = (exit_code == 124)
        analyze_output = r.stdout
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        timed_out = True

    stop_sampler.set()
    samp_thread.join(timeout=3)

    # ─── Save EXPLAIN ANALYZE result ───
    analyze_data = None
    if timed_out:
        analyze_data = {"qid": qid, "status": "timeout", "elapsed_s": round(elapsed, 1)}
    elif exit_code != 0:
        analyze_data = {"qid": qid, "status": "error", "elapsed_s": round(elapsed, 1)}
    else:
        # Include header for readability
        header = f"-- Query: {qid}\n-- SQL: {sql[:200]}...\n-- Execution time: {elapsed:.1f}s\n\n"
        analyze_data = {"qid": qid, "status": "ok", "elapsed_s": round(elapsed, 1),
                        "output": header + analyze_output}

    # ─── Compute cgroup metrics ───
    cgroup_data = None
    if exit_code == 0 and not timed_out:
        latency_s = round(elapsed, 3)

        after_net = read_network()
        after_disk = read_disk_io()

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

        cgroup_data = {
            "qid": qid,
            "status": "ok",
            "latency_s": latency_s,
            "memory_peak_bytes": {n: mem_peak.get(n, 0) for n in CONTAINERS},
            "memory_delta_bytes": mem_delta,
            "network_delta_bytes": net_delta,
            "disk_delta_bytes": disk_delta,
            "n_memory_samples": len(mem_samples),
        }
    elif timed_out:
        cgroup_data = {"qid": qid, "status": "timeout", "elapsed_s": round(elapsed, 1)}

    return analyze_data, cgroup_data


def save_result(result_file, qid, status):
    with open(result_file, "a") as f:
        f.write(f"{qid} {status}\n")


def load_done_queries(result_file):
    done = set()
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    done.add(parts[0])
    return done


def main():
    os.makedirs(ANALYZE_DIR, exist_ok=True)
    os.makedirs(CGROUP_DIR, exist_ok=True)

    # ─── Build query list from EXPLAIN plans ───
    all_queries = sorted([
        f.replace(".txt", "") for f in os.listdir(PLAN_DIR)
        if f.endswith(".txt") and f.replace(".txt", "").isdigit()
    ], key=int)

    # ─── Check what's already done ───
    # A query is "done" if it has BOTH EXPLAIN ANALYZE AND cgroup JSON
    done_analyze = set(f.replace(".txt", "") for f in os.listdir(ANALYZE_DIR) if f.endswith(".txt"))
    done_cgroup = set(f.replace(".json", "") for f in os.listdir(CGROUP_DIR) if f.endswith(".json"))
    done_both = done_analyze & done_cgroup

    pending = [q for q in all_queries if q not in done_both]
    only_analyze = [q for q in pending if q in done_analyze and q not in done_cgroup]
    only_cgroup = [q for q in pending if q in done_cgroup and q not in done_analyze]

    print(f"Combined EXPLAIN ANALYZE + cgroup collection")
    print(f"  Total plans:    {len(all_queries)}")
    print(f"  Both done:      {len(done_both)}")
    print(f"  Pending:        {len(pending)}")
    print(f"    - need ANALYZE only: {len(only_analyze)}")
    print(f"    - need cgroup only:  {len(only_cgroup)}")
    print(f"    - need both:         {len(pending) - len(only_analyze) - len(only_cgroup)}")
    print(f"  Timeout: {HARD_TIMEOUT_S}s per query")
    print(f"  Output: ANALYZE→{ANALYZE_DIR}  CGROUP→{CGROUP_DIR}\n")

    # Load result tracking
    done_tracking = load_done_queries(RESULT_FILE)

    n_ok = 0; n_fail = 0; n_timeout = 0; n_error = 0
    consecutive_fails = 0

    for i, qid in enumerate(pending):
        # Check TiDB health
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            if not check_tidb_alive():
                restart_tidb()
                consecutive_fails = 0
                time.sleep(RESTART_WAIT_S)

        need_analyze = qid not in done_analyze
        need_cgroup = qid not in done_cgroup

        print(f"[{i+1}/{len(pending)}] Q{qid} ...", end=" ", flush=True)
        analyze_data, cgroup_data = collect_one_query(qid)

        if analyze_data is None:
            print("FAIL (no SQL file)")
            save_result(RESULT_FILE, qid, "fail")
            n_fail += 1; consecutive_fails += 1
            continue

        status = analyze_data["status"]
        consecutive_fails = 0

        # ─── Save EXPLAIN ANALYZE ───
        if need_analyze:
            if status == "ok":
                with open(os.path.join(ANALYZE_DIR, f"{qid}.txt"), "w") as f:
                    f.write(analyze_data["output"])
            elif status == "timeout":
                with open(os.path.join(ANALYZE_DIR, f"{qid}.txt"), "w") as f:
                    f.write(f"-- Query {qid}: TIMEOUT ({analyze_data['elapsed_s']:.0f}s)\n")
                n_timeout += 1
            else:
                with open(os.path.join(ANALYZE_DIR, f"{qid}.txt"), "w") as f:
                    f.write(f"-- Query {qid}: ERROR\n")
                n_error += 1

        # ─── Save cgroup ───
        if need_cgroup and cgroup_data is not None:
            with open(os.path.join(CGROUP_DIR, f"{qid}.json"), "w") as f:
                json.dump(cgroup_data, f)

        # ─── Status reporting ───
        if status == "ok":
            net_t = sum(cgroup_data.get("network_delta_bytes", {}).values()) if cgroup_data else 0
            mem_t = sum(cgroup_data.get("memory_delta_bytes", {}).values()) if cgroup_data else 0
            disk_t = sum(cgroup_data.get("disk_delta_bytes", {}).values()) if cgroup_data else 0
            def fmt(b):
                if b > 1e9: return f"{b/1e9:.1f}G"
                if b > 1e6: return f"{b/1e6:.1f}M"
                return f"{b/1e3:.1f}K"
            print(f"OK {analyze_data['elapsed_s']:.0f}s m={fmt(mem_t)} n={fmt(net_t)} d={fmt(disk_t)}")
            save_result(RESULT_FILE, qid, "ok")
            n_ok += 1
        elif status == "timeout":
            print(f"TIMEOUT ({analyze_data['elapsed_s']:.0f}s)")
            save_result(RESULT_FILE, qid, "timeout")
            n_timeout += 1
            consecutive_fails += 1
        else:
            print("ERROR")
            save_result(RESULT_FILE, qid, "error")
            n_error += 1
            consecutive_fails += 1

        # ─── Periodic progress ───
        if (n_ok + n_fail + n_timeout + n_error) % 50 == 0:
            print(f"  --- progress: {n_ok} ok, {n_timeout} timeout, {n_error} error, {n_fail} fail ---")

    # ─── Summary ───
    print(f"\nDone! OK={n_ok} TIMEOUT={n_timeout} ERROR={n_error} FAIL={n_fail}")


if __name__ == "__main__":
    main()
