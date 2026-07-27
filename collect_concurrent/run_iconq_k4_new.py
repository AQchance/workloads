#!/usr/bin/env python3
"""
ICONQ-style scheduler — real execution, K=4, Bi-LSTM slowdown prediction.
Uses arrival_times_3r_poisson.csv, same crash recovery as FIFO scheduler.
"""
import subprocess, time, threading, os, csv, json, math, sys
import numpy as np
import torch, torch.nn as nn
from collections import deque

sys.path.insert(0, '/home/anqian/Desktop/my_lab/workloads/lstm')

ARRIVAL_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/arrival_times_3r_poisson.csv'
SQL_DIR     = '/home/anqian/Desktop/my_lab/workloads/SQLStorm'
RESOURCE_CACHE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/tabpfn_258_predictions_oof.json'
MODEL_PATH  = '/home/anqian/Desktop/my_lab/workloads/final_queries/bilstm_tabpfn.pt'
NORM_PATH   = '/home/anqian/Desktop/my_lab/workloads/final_queries/bilstm_tabpfn_norm.npz'
OUT_CSV     = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/iconq_k4_new_trace.csv'
CKPT_FILE   = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/iconq_k4_new_checkpoint.json'

K, TIMEOUT_S, COOLDOWN_S = 4, 600, 30
DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']
MYSQL_CMD = ['mysql','-h','172.19.0.11','-P','4000','-u','root','-D','tpch_sf40']


# ═══════════ Bi-LSTM Model ═══════════
class BiLSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
    def forward(self, X, lengths):
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


# ═══════════ Infrastructure ═══════════
def load_arrivals():
    arr = []
    with open(ARRIVAL_FILE) as f:
        for row in csv.DictReader(f): arr.append((row['qid'], float(row['arrival_time_s'])))
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


# ═══════════ Slowdown Predictor ═══════════
class SlowdownPredictor:
    def __init__(self):
        with open(RESOURCE_CACHE) as f: self.resources = json.load(f)
        norm = np.load(NORM_PATH)
        self.X_mean, self.X_std = norm['X_mean'], norm['X_std']
        self.y_mean, self.y_std = float(norm['y_mean']), float(norm['y_std'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BiLSTM(input_dim=17)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()

    @staticmethod
    def _conflict(t, c):
        t_arr, c_arr = np.array(t), np.array(c)
        return list(np.minimum(t_arr, c_arr) / np.maximum(np.abs(t_arr)+np.abs(c_arr)+1e-8, 1e-8))

    def predict_slowdown(self, target_qid, running_qids):
        if target_qid not in self.resources: return 1.5
        pred_i = self.resources[target_qid]
        qv = [pred_i[d] for d in DIMS]; tr_ = [pred_i[d] for d in DIMS]
        seq = []
        for oq in running_qids:
            if oq not in self.resources: continue
            pred_j = self.resources[oq]
            ovv = [pred_j[d] for d in DIMS]; oc = [pred_j[d] for d in DIMS]
            c = self._conflict(tr_, oc)
            seq.append(qv + ovv + [0.0, 0.0] + c)  # time_diff=0, is_before=0
        if not seq: return 1.0
        X = np.stack(seq, dtype=np.float32)
        X = (X - self.X_mean) / self.X_std
        X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
        L = torch.LongTensor([len(seq)]).to(self.device)
        with torch.no_grad():
            pred_z = self.model(X_t, L).cpu().item()
        ratio = max(np.exp(pred_z * self.y_std + self.y_mean) - 1, 0.01)
        return ratio


# ═══════════ Checkpoint ═══════════
def save_checkpoint(elapsed, aidx, qlist, results):
    ck = {'elapsed_s': round(elapsed,1), 'arrival_idx': aidx, 'queue': qlist, 'results': results,
          'n_ok': sum(1 for r in results if r['status']=='ok'), 'n_oom': sum(1 for r in results if r['status']=='oom')}
    with open(CKPT_FILE, 'w') as f: json.dump(ck, f)

def load_checkpoint():
    if not os.path.exists(CKPT_FILE): return None
    with open(CKPT_FILE) as f: return json.load(f)


# ═══════════ Main ═══════════
def main():
    arrivals = load_arrivals()
    N = len(arrivals)
    predictor = SlowdownPredictor()
    print(f"ICONQ K={K} | {N} queries | max arrival {arrivals[-1][1]/3600:.1f}h | timeout={TIMEOUT_S}s")
    print(f"Bi-LSTM: {sum(p.numel() for p in predictor.model.parameters()):,} params, {len(predictor.resources)} resources")

    ck = load_checkpoint()
    if ck and ck.get('results'):
        print(f"RESUMING: {len(ck['results'])} done, fed={ck['arrival_idx']}, elapsed={ck['elapsed_s']:.0f}s")
        results, aidx_start, queue_list, elapsed_offset = ck['results'], ck['arrival_idx'], deque(ck['queue']), ck['elapsed_s']
    else:
        print("Fresh start")
        results, aidx_start, queue_list, elapsed_offset = [], 0, deque(), 0.0

    queue = deque(queue_list)
    lock, stop = threading.Lock(), threading.Event()
    running = {}
    running_lock = threading.Lock()
    res_lock = threading.Lock()
    aidx = [aidx_start]
    gstart = [time.time()]
    consecutive_ooms = [0]
    last_ckpt_n = [len(results)]

    THETA = 1.5  # slowdown threshold for submission
    alpha = 0.95  # wait-time decay factor for starvation avoidance

    def worker(wid):
        while not stop.is_set():
            with lock:
                if not queue: time.sleep(0.5); continue
                # ICONQ greedy with starvation avoidance
                with running_lock: rqids = list(running.keys())
                now_wall = time.time() - gstart[0] + elapsed_offset
                best_qid, best_arr, best_idx = None, None, -1
                best_score = float('inf')
                best_pred_slow = 0.0
                best_raw_score = 0.0

                for idx in range(len(queue)):
                    qid, arr_t = queue[idx]
                    # Part A: predicted slowdown of candidate itself
                    pred_slow = predictor.predict_slowdown(qid, rqids)
                    # Part B: additional slowdown inflicted on each running query
                    delta_running = 0.0
                    for rq in rqids:
                        s_before = predictor.predict_slowdown(rq, [x for x in rqids if x != rq])
                        s_after  = predictor.predict_slowdown(rq, [x for x in rqids if x != rq] + [qid])
                        delta_running += max(s_after - s_before, 0.0)
                    raw_score = pred_slow + delta_running
                    # Starvation avoidance
                    wait_time = max(0, now_wall - arr_t)
                    score = raw_score * (alpha ** (wait_time / 10.0))
                    if score < best_score:
                        best_score, best_qid, best_arr, best_idx = score, qid, arr_t, idx
                        best_pred_slow = pred_slow
                        best_raw_score = raw_score

                if best_qid is None:
                    best_qid, best_arr = queue.popleft()
                    best_score = 0
                else:
                    del queue[best_idx]

            with running_lock: running[best_qid] = time.time()
            sw = time.time(); status, rt = execute_query(best_qid); fw = time.time()
            with running_lock: running.pop(best_qid, None)
            if status == 'timeout': rt = TIMEOUT_S

            with res_lock:
                results.append({
                    'qid': best_qid, 'arrival': best_arr,
                    'start': round(sw-gstart[0]+elapsed_offset, 1),
                    'finish': round(fw-gstart[0]+elapsed_offset, 1) if status=='ok' else None,
                    'runtime': round(rt, 2), 'status': status,
                    'predicted_slowdown': round(best_score, 3),
                })
            # Append CSV
            with open(OUT_CSV, 'a', newline='') as _tf:
                _w = csv.writer(_tf)
                if _tf.tell() == 0: _w.writerow(['qid','arrival','start','finish','runtime','status','slowdown_pred','raw_score','predicted_slowdown'])
                _w.writerow([best_qid, best_arr, round(sw-gstart[0]+elapsed_offset,1),
                            round(fw-gstart[0]+elapsed_offset,1) if status=='ok' else '',
                            round(rt,2), status, round(best_pred_slow,3), round(best_raw_score,3), round(best_score,3)])

            if status == 'oom': consecutive_ooms[0] += 1
            elif status == 'ok': consecutive_ooms[0] = 0

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(K)]
    for t in threads: t.start()

    try:
        while aidx[0] < N:
            elapsed = time.time()-gstart[0]+elapsed_offset
            nr = len(results)
            if nr-last_ckpt_n[0] >= 25:
                with lock: ql = list(queue)
                with res_lock: rs = list(results)
                save_checkpoint(elapsed, aidx[0], ql, rs)
                last_ckpt_n[0] = nr
                n_ok = sum(1 for r in rs if r['status']=='ok')
                n_oom = sum(1 for r in rs if r['status']=='oom')
                print(f"  [{elapsed:.0f}s] fed={aidx[0]}/{N}, ok={n_ok}, oom={n_oom}, queue={len(ql)} 💾")

            if consecutive_ooms[0] >= 3:
                print(f"\n  *** {consecutive_ooms[0]} consecutive OOMs — restarting ***")
                with lock: ql = list(queue)
                with res_lock: rs = list(results)
                save_checkpoint(elapsed, aidx[0], ql, rs)
                restart_cluster()
                gstart[0] = time.time(); consecutive_ooms[0] = 0
                continue

            with lock: qs = len(queue)
            if qs == 0 and aidx[0] < N:
                next_arr = arrivals[aidx[0]][1]
                if next_arr > elapsed+5:
                    elapsed_offset += next_arr-elapsed
                    gstart[0] = time.time()
                    continue

            while aidx[0] < N:
                qid, arr_t = arrivals[aidx[0]]
                if arr_t <= elapsed:
                    with lock: queue.append((qid, arr_t))
                    aidx[0] += 1
                else: break
            time.sleep(0.3)

        print(f"\n  All fed. Draining {len(queue)} remaining...")
        while True:
            with lock: qs = len(queue)
            nr2 = len(results)
            if qs == 0 and nr2 >= N: break
            if nr2-last_ckpt_n[0] >= 25:
                elapsed = time.time()-gstart[0]+elapsed_offset
                with lock: ql = list(queue)
                with res_lock: rs = list(results)
                save_checkpoint(elapsed, aidx[0], ql, rs)
                last_ckpt_n[0] = nr2
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        elapsed = time.time()-gstart[0]+elapsed_offset
        with lock: ql = list(queue)
        with res_lock: rs = list(results)
        save_checkpoint(elapsed, aidx[0], ql, rs)
    finally:
        stop.set()
        for t in threads: t.join(timeout=10)

    elapsed = time.time()-gstart[0]+elapsed_offset
    with lock: ql = list(queue)
    with res_lock: rs = list(results)
    save_checkpoint(elapsed, aidx[0], ql, rs)

    n_ok = sum(1 for r in results if r['status']=='ok')
    n_oom = sum(1 for r in results if r['status']=='oom')
    print(f"\nDone: {len(results)} results — {n_ok} ok, {n_oom} oom")
    completed = [r for r in results if r.get('finish') is not None]
    if completed:
        lats = sorted([r['finish']-r['arrival'] for r in completed])
        print(f"E2E: mean={sum(lats)/len(lats):.0f}s P50={lats[len(lats)//2]:.0f}s P90={lats[9*len(lats)//10]:.0f}s P99={lats[99*len(lats)//100]:.0f}s")
    print(f"Saved: {OUT_CSV}")


if __name__ == '__main__': main()
