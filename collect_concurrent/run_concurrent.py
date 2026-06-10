#!/usr/bin/env python3
"""
Concurrent query execution, K=2, 3 rounds of 1000 queries each.
Round 1: only remaining 644 un-executed queries (appends to existing trace)
Rounds 2-3: full 1000 queries each
"""

import subprocess, time, threading, os, random

QUERY_DIR = '/home/anqian/Desktop/my_lab/workloads/SQLStorm'
QUERY_ALL = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries.txt'
QUERY_R1  = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries.txt'
OUT_DIR   = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent'
MYSQL_CMD = ['mysql', '-h', '172.19.0.11', '-P', '4000', '-u', 'root', '-D', 'tpch_sf40']

NUM_CLIENTS = 4
TOTAL_ROUNDS = 1         # 1 round = 1000 queries
TIMEOUT_S = 600
PENALTY_S = 600
COOLDOWN_S = 30
STAGGER_COUNT = 8        # first 8 queries staggered 5s apart
STAGGER_DELAY = 5        # seconds between staggered starts

start_time = 0.0
csv_file = None
pool_lock = threading.Lock()
running_lock = threading.Lock()
running_queries = {}
query_pool = []
stop_flag = threading.Event()
result_count = [0, 0, 0]  # ok, error, penalty


def check_tidb():
    try:
        r = subprocess.run(MYSQL_CMD + ['-e', 'SELECT COUNT(*) FROM supplier'],
                          capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except:
        return False


def restart_tidb():
    print("\n  *** RESTARTING TiDB ***")
    try:
        subprocess.run(['docker', 'exec', 'tidb1', 'bash', '/root/tidb_start.sh'],
                      capture_output=True, text=True, timeout=120)
    except: pass
    for a in range(40):
        if check_tidb():
            print(f"  *** Ready (attempt {a+1}) ***\n")
            time.sleep(COOLDOWN_S)
            return True
        time.sleep(3)
    return False


def save_result(qid, start, runtime, status):
    global csv_file
    if csv_file:
        csv_file.write(f"{qid},{start},{runtime},{status}\n")
        csv_file.flush()
    if status == 'ok': result_count[0] += 1
    elif status == 'penalty': result_count[2] += 1
    else: result_count[1] += 1


def execute_query(qid):
    sf = os.path.join(QUERY_DIR, f'{qid}.sql')
    if not os.path.exists(sf): return False, 0.0
    with open(sf) as f: sql = f.read().strip().rstrip(';')
    t0 = time.time()
    try:
        r = subprocess.run(MYSQL_CMD + ['-e', sql], capture_output=True, text=True, timeout=TIMEOUT_S)
        return r.returncode == 0, time.time() - t0
    except: return False, time.time() - t0


def worker(wid):
    global query_pool
    # Staggered start: first 8 workers start 5s apart to avoid crashing TiDB
    if wid < STAGGER_COUNT:
        time.sleep(wid * STAGGER_DELAY)

    while not stop_flag.is_set():
        with pool_lock:
            if len(query_pool) == 0:
                return  # round done
            qid = query_pool.pop()

        if not check_tidb():
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] TiFlash down, waiting...")
            while not check_tidb(): time.sleep(10)
            with pool_lock: query_pool.append(qid)
            continue

        with running_lock: running_queries[wid] = qid
        elapsed = time.time() - start_time
        success, runtime = execute_query(qid)
        with running_lock: running_queries.pop(wid, None)

        if success:
            save_result(qid, round(elapsed, 1), round(runtime, 2), 'ok')
            if result_count[0] % 100 == 0:
                print(f"  [{elapsed:.0f}s] {result_count[0]} ok, {result_count[2]} penalty")
        else:
            save_result(qid, round(elapsed, 1), PENALTY_S, 'penalty')
            print(f"  [{elapsed:.0f}s] Q{qid}: ERR → penalty, restarting")
            restart_tidb()
        time.sleep(0.3)


def run_round(label, queries):
    global start_time, query_pool, stop_flag
    query_pool = queries[:]
    random.shuffle(query_pool)
    print(f"\n=== {label} ({len(query_pool)} queries) ===")

    stop_flag.clear()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(NUM_CLIENTS)]
    for t in threads: t.start()

    while True:
        time.sleep(5)
        with pool_lock: pe = len(query_pool)
        with running_lock: nr = len(running_queries)
        if pe == 0 and nr == 0: break

    stop_flag.set()
    for t in threads: t.join(timeout=10)
    print(f"  {label} done: {result_count[0]} ok, {result_count[2]} penalty\n")


def main():
    global start_time, csv_file

    with open(QUERY_ALL) as f: all_q = [l.strip() for l in f if l.strip()]
    with open(QUERY_R1) as f: r1_q = [l.strip() for l in f if l.strip()]

    out_path = os.path.join(OUT_DIR, f'trace_{NUM_CLIENTS}.csv')
    csv_file = open(out_path, 'a')
    if os.path.getsize(out_path) == 0:
        csv_file.write("qid,start,runtime,status\n")
        csv_file.flush()

    print(f"K={NUM_CLIENTS} | All={len(all_q)} | R1 remaining={len(r1_q)} | Append mode\n")

    while not check_tidb():
        print(f"Waiting TiFlash... {time.strftime('%H:%M:%S')}")
        time.sleep(30)

    start_time = time.time()

    run_round("R1: finish remaining 1st pass", r1_q)
    run_round("R2: full 1000", all_q)
    run_round("R3: full 1000", all_q)

    csv_file.close()
    print(f"ALL DONE: {result_count[0]} ok, {result_count[2]} penalty")


if __name__ == '__main__':
    main()
