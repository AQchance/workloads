#!/usr/bin/env python3
"""
FIFO scheduler, Poisson arrivals, K=2, checkpoint/resume.
"""
import subprocess, time, threading, os, csv, json
from collections import deque

ARRIVAL_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/arrival_times_poisson.csv'
SQL_DIR     = '/home/anqian/Desktop/my_lab/workloads/SQLStorm'
OUT_CSV     = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/fifo_k2_trace.csv'
CKPT_FILE   = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/fifo_k2_checkpoint.json'
K, TIMEOUT_S, COOLDOWN_S = 2, 600, 30
MYSQL_CMD = ['mysql','-h','172.19.0.11','-P','4000','-u','root','-D','tpch_sf40']

def load_arrivals():
    arr=[]
    with open(ARRIVAL_FILE) as f:
        for row in csv.DictReader(f): arr.append((row['qid'],float(row['arrival_time_s'])))
    return arr

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
    sf=os.path.join(SQL_DIR,f'{qid}.sql')
    if not os.path.exists(sf): return 'error',0.0
    with open(sf) as f: sql=f.read().strip().rstrip(';')
    try:
        t0=time.time()
        r=subprocess.run(MYSQL_CMD+['-e',sql],capture_output=True,text=True,timeout=TIMEOUT_S)
        dt=time.time()-t0
        if r.returncode!=0:
            el=r.stderr.lower()
            if any(k in el for k in ['out of memory','memory exceeded','memory limit','server has gone away','lost connection',"can't connect",'error 2003','error 2006','error 2013']): return 'oom',dt
            return 'error',dt
        return 'ok',dt
    except subprocess.TimeoutExpired: return 'timeout',TIMEOUT_S
    except: return 'error',0.0

def save_checkpoint(elapsed,aidx,qlist,results):
    ck={'elapsed_s':round(elapsed,1),'arrival_idx':aidx,'queue':qlist,'results':results,
        'n_ok':sum(1 for r in results if r['status']=='ok'),'n_oom':sum(1 for r in results if r['status']=='oom')}
    with open(CKPT_FILE,'w') as f: json.dump(ck,f)
    if results:
        with open(OUT_CSV,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['qid','arrival','start','finish','runtime','status']); w.writeheader()
            for r in results: w.writerow(r)

def load_checkpoint():
    if not os.path.exists(CKPT_FILE): return None
    with open(CKPT_FILE) as f: return json.load(f)

def main():
    arrivals = load_arrivals()
    N = len(arrivals)
    print(f"Poisson mean=30s | K={K} | {N} queries | max arrival {arrivals[-1][1]/3600:.1f}h | timeout={TIMEOUT_S}s")

    ck = load_checkpoint()
    if ck and ck.get('results'):
        nprev = len(ck['results'])
        print(f"RESUMING: {nprev} done, fed={ck['arrival_idx']}, elapsed={ck['elapsed_s']:.0f}s")
        results, aidx_start, queue, elapsed_offset = ck['results'], ck['arrival_idx'], deque(ck['queue']), ck['elapsed_s']
    else:
        print("Fresh start")
        results, aidx_start, queue, elapsed_offset = [], 0, deque(), 0.0

    lock, res_lock, stop = threading.Lock(), threading.Lock(), threading.Event()
    aidx = [aidx_start]
    gstart = [time.time()]
    consecutive_ooms = [0]
    last_ckpt_n = [len(results)]

    def worker(_):
        while not stop.is_set():
            with lock:
                if not queue: time.sleep(0.5); continue
                qid, arr_t = queue.popleft()
            sw=time.time(); status,rt=execute_query(qid); fw=time.time()
            if status=='timeout': rt=TIMEOUT_S
            with res_lock:
                results.append({'qid':qid,'arrival':arr_t,
                    'start':round(sw-gstart[0]+elapsed_offset,1),
                    'finish':round(fw-gstart[0]+elapsed_offset,1) if status=='ok' else None,
                    'runtime':round(rt,2),'status':status})
            if status=='oom': consecutive_ooms[0]+=1
            elif status=='ok': consecutive_ooms[0]=0
            # Append to CSV immediately
            with open('/home/anqian/Desktop/my_lab/workloads/collect_concurrent/fifo_k2_trace.csv','a',newline='') as _tf:
                import csv as _csv
                _w=_csv.writer(_tf)
                if _tf.tell()==0: _w.writerow(['qid','arrival','start','finish','runtime','status'])
                _w.writerow([qid,arr_t,round(sw-gstart[0]+elapsed_offset,1),round(fw-gstart[0]+elapsed_offset,1) if status=='ok' else '',round(rt,2),status])

    threads=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(K)]
    for t in threads: t.start()

    try:
        while aidx[0] < N:
            elapsed = time.time()-gstart[0]+elapsed_offset

            # Periodic checkpoint
            nr = len(results)
            if nr-last_ckpt_n[0]>=25:
                with lock: ql=list(queue)
                with res_lock: rs=list(results)
                save_checkpoint(elapsed,aidx[0],ql,rs)
                last_ckpt_n[0]=nr
                n_ok=sum(1 for r in results if r['status']=='ok')
                n_oom=sum(1 for r in results if r['status']=='oom')
                print(f"  [{elapsed:.0f}s] fed={aidx[0]}/{N}, ok={n_ok}, oom={n_oom}, queue={len(ql)} 💾")

            # OOM cascade → restart + resume
            if consecutive_ooms[0]>=3:
                print(f"\n  *** {consecutive_ooms[0]} consecutive OOMs — restarting ***")
                with lock: ql=list(queue)
                with res_lock: rs=list(results)
                save_checkpoint(elapsed,aidx[0],ql,rs)
                restart_cluster()
                gstart[0]=time.time(); consecutive_ooms[0]=0
                continue

            # Fast-forward if idle
            with lock: qs=len(queue)
            if qs==0 and aidx[0]<N:
                next_arr=arrivals[aidx[0]][1]
                if next_arr>elapsed+5:
                    elapsed_offset+=next_arr-elapsed
                    gstart[0]=time.time()
                    continue

            # Feed arrivals
            while aidx[0]<N:
                qid,arr_t=arrivals[aidx[0]]
                if arr_t<=elapsed:
                    with lock: queue.append((qid,arr_t))
                    aidx[0]+=1
                else: break
            time.sleep(0.3)

        # Drain
        print(f"\n  All fed. Draining {len(queue)} remaining...")
        while True:
            with lock: qs=len(queue)
            nr2=len(results)
            if qs==0 and nr2>=N: break
            if nr2-last_ckpt_n[0]>=25:
                elapsed=time.time()-gstart[0]+elapsed_offset
                with lock: ql=list(queue)
                with res_lock: rs=list(results)
                save_checkpoint(elapsed,aidx[0],ql,rs)
                last_ckpt_n[0]=nr2
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        elapsed=time.time()-gstart[0]+elapsed_offset
        with lock: ql=list(queue)
        with res_lock: rs=list(results)
        save_checkpoint(elapsed,aidx[0],ql,rs)
    finally:
        stop.set()
        for t in threads: t.join(timeout=10)

    elapsed=time.time()-gstart[0]+elapsed_offset
    with lock: ql=list(queue)
    with res_lock: rs=list(results)
    save_checkpoint(elapsed,aidx[0],ql,rs)
    n_ok=sum(1 for r in results if r['status']=='ok'); n_oom=sum(1 for r in results if r['status']=='oom')
    print(f"\nDone: {len(results)} results — {n_ok} ok, {n_oom} oom")
    completed=[r for r in results if r['finish'] is not None]
    if completed:
        lats=sorted([r['finish']-r['arrival'] for r in completed])
        print(f"E2E: mean={sum(lats)/len(lats):.0f}s P50={lats[len(lats)//2]:.0f}s P90={lats[9*len(lats)//10]:.0f}s P99={lats[99*len(lats)//100]:.0f}s")
    print(f"Saved: {OUT_CSV}")

if __name__=='__main__': main()
