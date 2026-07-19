#!/usr/bin/env python3
"""K=4 batch executor, resume from checkpoint, time offset."""
import subprocess, time, threading, os, csv, json

ORDER_FILE = '/home/anqian/Desktop/my_lab/workloads/final_queries/query_order_10r.txt'
SQL_DIR    = '/home/anqian/Desktop/my_lab/workloads/SQLStorm'
OUT_CSV    = '/home/anqian/Desktop/my_lab/workloads/final_queries/k8_batch_trace.csv'
CKPT_FILE  = '/home/anqian/Desktop/my_lab/workloads/final_queries/k8_batch_checkpoint.json'
K, TIMEOUT_S, COOLDOWN_S = 8, 600, 30
MYSQL_CMD = ['mysql','-h','172.19.0.11','-P','4000','-u','root','-D','tpch_sf40']

def load_queries():
    qs = []
    with open(ORDER_FILE) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2: qs.append(parts[1])
    return qs

def check_tidb():
    try: return subprocess.run(MYSQL_CMD+['-e','SELECT 1'],capture_output=True,text=True,timeout=10).returncode==0
    except: return False

def restart_cluster():
    print("\n  *** RESTARTING ***")
    subprocess.run(['bash','/home/anqian/Desktop/my_lab/docker_start.sh'],capture_output=True,timeout=120)
    time.sleep(5)
    subprocess.run(['docker','exec','tidb1','bash','/root/tidb_start.sh'],capture_output=True,timeout=120)
    for a in range(40):
        if check_tidb(): print(f"  Ready (attempt {a+1})\n"); time.sleep(COOLDOWN_S); return True
        time.sleep(3)
    return False

def execute_query(qid):
    sf = os.path.join(SQL_DIR, f'{qid}.sql')
    if not os.path.exists(sf): return 'error', 0.0
    with open(sf) as f: sql = f.read().strip().rstrip(';')
    try:
        t0 = time.time()
        r = subprocess.run(MYSQL_CMD+['-e',sql],capture_output=True,text=True,timeout=TIMEOUT_S)
        dt = time.time()-t0
        if r.returncode != 0:
            el = r.stderr.lower()
            if any(k in el for k in ['out of memory','memory exceeded','memory limit','server has gone away','lost connection',"can't connect",'error 2003','error 2006','error 2013']):
                return 'oom', dt
            return 'error', dt
        return 'ok', dt
    except subprocess.TimeoutExpired: return 'timeout', TIMEOUT_S
    except: return 'error', 0.0

def save_checkpoint(next_idx, results):
    ck = {'next_idx': next_idx, 'results': results,
          'n_ok': sum(1 for r in results if r['status']=='ok'),
          'n_oom': sum(1 for r in results if r['status']=='oom')}
    with open(CKPT_FILE,'w') as f: json.dump(ck, f)

def load_checkpoint():
    if not os.path.exists(CKPT_FILE): return None
    with open(CKPT_FILE) as f: return json.load(f)

def main():
    queries = load_queries(); N = len(queries)
    print(f"K={K} BATCH | {N} queries | timeout={TIMEOUT_S}s")

    ck = load_checkpoint()
    if ck and ck.get('next_idx', 0) > 0:
        idx_start = ck['next_idx']
        results_run = ck.get('results', [])
        # Get time offset: last finish time from CSV
        t_offset = 0.0
        if os.path.exists(OUT_CSV):
            with open(OUT_CSV) as f:
                reader = list(csv.DictReader(f))
                if reader:
                    finishes = [float(r['finish']) for r in reader if r['finish']]
                    if finishes: t_offset = max(finishes)
        print(f"RESUMING at idx={idx_start}, time_offset={t_offset:.0f}s, {len(results_run)} prev results")
    else:
        idx_start, results_run, t_offset = 0, [], 0.0
        print("Fresh start")

    idx = [idx_start]
    results_this = results_run[:]
    lock = threading.Lock(); stop = threading.Event()
    consecutive_ooms = [0]
    gstart = time.time()
    last_ckpt_n = [len(results_this)]

    def worker(wid):
        while not stop.is_set():
            with lock:
                if idx[0] >= N: time.sleep(0.5); continue
                qid = queries[idx[0]]; idx[0] += 1
            sw = time.time(); status, rt = execute_query(qid); fw = time.time()
            if status == 'timeout': rt = TIMEOUT_S
            with lock:
                results_this.append({
                    'qid': qid,
                    'start': round(sw - gstart + t_offset, 1),
                    'finish': round(fw - gstart + t_offset, 1) if status=='ok' else None,
                    'runtime': round(rt, 2), 'status': status,
                })
            # Append to CSV
            with open(OUT_CSV, 'a', newline='') as _tf:
                _w = csv.writer(_tf)
                if _tf.tell() == 0: _w.writerow(['qid','start','finish','runtime','status'])
                _w.writerow([qid, round(sw-gstart+t_offset,1),
                            round(fw-gstart+t_offset,1) if status=='ok' else '',
                            round(rt,2), status])
            if status == 'oom':
                consecutive_ooms[0] += 1
                time.sleep(30)  # cooldown after OOM
            elif status == 'ok': consecutive_ooms[0] = 0

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(K)]
    for t in threads: t.start()

    try:
        while True:
            time.sleep(3)
            nr = len(results_this)
            if nr - last_ckpt_n[0] >= 25:
                save_checkpoint(idx[0], results_this)
                last_ckpt_n[0] = nr
                n_ok = sum(1 for r in results_this if r['status']=='ok')
                n_oom = sum(1 for r in results_this if r['status']=='oom')
                elapsed = time.time() - gstart
                print(f"  [{elapsed:.0f}s] {idx[0]}/{N} processed, ok={n_ok}, oom={n_oom} 💾")
            if consecutive_ooms[0] >= 3:
                print(f"\n  *** {consecutive_ooms[0]} consecutive OOMs — restarting ***")
                save_checkpoint(idx[0], results_this)
                restart_cluster(); gstart = time.time(); consecutive_ooms[0] = 0
                continue
            if idx[0] >= N: break
    except KeyboardInterrupt:
        print("\nInterrupted — checkpoint saved"); save_checkpoint(idx[0], results_this)
    finally:
        stop.set()
        for t in threads: t.join(timeout=10)

    save_checkpoint(idx[0], results_this)
    n_ok = sum(1 for r in results_this if r['status']=='ok')
    print(f"\nDone: {len(results_this)} total — {n_ok} ok")
    print(f"Saved: {OUT_CSV}")

if __name__ == '__main__': main()
