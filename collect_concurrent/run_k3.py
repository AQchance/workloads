#!/usr/bin/env python3
"""K=3 concurrent collector with auto-restart, resume support, staggered start."""

import subprocess, time, threading, os

QUERY_DIR = '/home/anqian/Desktop/my_lab/workloads/SQLStorm'
ORDER_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/query_order_k3_v2.txt'
OUT_CSV  = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_3.csv'
MYSQL_CMD = ['mysql', '-h', '172.19.0.11', '-P', '4000', '-u', 'root', '-D', 'tpch_sf40']

K = 3
STAGGER_S = 3          # first K queries after restart staggered 3s apart
TIMEOUT_S = 600
PENALTY_S = 600
COOLDOWN_S = 30

start_time = 0.0
csv_file = None
order_lock = threading.Lock()
running_lock = threading.Lock()
running_queries = {}
query_order = []
next_idx = 0
restart_event = threading.Event()
result_count = [0, 0, 0]
stop_flag = threading.Event()


def check_tidb():
    try:
        r = subprocess.run(MYSQL_CMD + ['-e', 'SELECT COUNT(*) FROM supplier'],
                          capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except: return False


def restart_tidb():
    print("\n  *** RESTARTING TiDB ***")
    try:
        subprocess.run(['docker', 'exec', 'tidb1', 'bash', '/root/tidb_start.sh'],
                      capture_output=True, text=True, timeout=120)
    except: pass
    for a in range(40):
        if check_tidb(): print(f"  *** Ready (attempt {a+1}) ***\n"); time.sleep(COOLDOWN_S); return True
        time.sleep(3)
    return False


def save_result(qid, start, runtime, status):
    if csv_file:
        csv_file.write(f"{qid},{start},{runtime},{status}\n"); csv_file.flush()
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
    global next_idx
    while not stop_flag.is_set():
        if restart_event.is_set() and wid > 0:
            time.sleep(wid * STAGGER_S)

        with order_lock:
            if next_idx >= len(query_order): restart_event.clear(); return
            qid = query_order[next_idx]; next_idx += 1
        if wid == 0: restart_event.clear()

        if not check_tidb():
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] TiFlash down, waiting...")
            while not check_tidb(): time.sleep(10)
            with order_lock: next_idx -= 1
            restart_event.set(); continue

        with running_lock: running_queries[wid] = qid
        elapsed = time.time() - start_time
        success, runtime = execute_query(qid)
        with running_lock: running_queries.pop(wid, None)

        if success:
            save_result(qid, round(elapsed, 1), round(runtime, 2), 'ok')
            if result_count[0] % 50 == 0: print(f"  [{elapsed:.0f}s] {result_count[0]} ok, {result_count[2]} penalty")
        else:
            save_result(qid, round(elapsed, 1), PENALTY_S, 'penalty')
            print(f"  [{elapsed:.0f}s] Q{qid}: ERR -> penalty, restarting")
            restart_tidb(); restart_event.set()
            with order_lock: next_idx -= 1
        time.sleep(0.3)


def main():
    global start_time, query_order, next_idx, csv_file

    with open(ORDER_FILE) as f: query_order = [l.strip() for l in f if l.strip()]

    csv_file = open(OUT_CSV, 'a')
    if os.path.getsize(OUT_CSV) == 0: csv_file.write("qid,start,runtime,status\n"); csv_file.flush()

    next_idx = 0  # fresh start
    print(f"K={K} | Total={len(query_order)} | Fresh start | Append mode\n")

    while not check_tidb():
        print(f"Waiting TiFlash... {time.strftime('%H:%M:%S')}"); time.sleep(30)

    start_time = time.time(); restart_event.set()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(K)]
    for t in threads: t.start()

    while not stop_flag.is_set():
        time.sleep(10)
        with order_lock:
            if next_idx >= len(query_order):
                with running_lock:
                    if len(running_queries) == 0: stop_flag.set()

    for t in threads: t.join(timeout=30); csv_file.close()
    print(f"\nDone: {result_count[0]} ok, {result_count[2]} penalty")


if __name__ == '__main__': main()
