"""
TabPFN-Scheduler — ICONQ greedy algorithm + TabPFN→Bi-LSTM predictor.

Stage 1: TabPFN OOF resource predictions (pre-cached, zero leak)
Stage 2: Bi-LSTM slowdown prediction (trained via train_bilstm_tabpfn.py)
Scheduling: ICONQ-style event-driven greedy

Feature encoding: 17-dim interaction vector
  Target:   mem, disk, net, lat, cpures                     =  5
  Peer:     mem, disk, net, lat, cpures                     =  5
  Timing:   start_diff, is_before                           =  2
  Conflict: min(t,c)/(|t|+|c|) × 5 resources               =  5
                                                     Total = 17

Usage:
  python collect_concurrent/tabpfn_scheduler.py --trace trace_2_mixed
"""
import os, json, csv, math, numpy as np, torch, torch.nn as nn
from collections import deque
from typing import Dict, List, Tuple, Optional

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')

# Paths
RESOURCE_CACHE = os.path.join(ROOT, 'collect_concurrent',
                               'tabpfn_258_predictions_oof.json')
MODEL_PATH = os.path.join(CKPT_DIR, 'bilstm_tabpfn.pt')
NORM_PATH = os.path.join(CKPT_DIR, 'bilstm_tabpfn_norm.npz')

DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']
MAX_CONCURRENT = 10
THETA = 1.5  # slowdown threshold for "beneficial" check


# ═══════════ Bi-LSTM Model (same as train_bilstm_tabpfn.py) ═══════════

class BiLSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


# ═══════════ Slowdown Predictor ═══════════

class SlowdownPredictor:
    """Predicts slowdown ratio using trained Bi-LSTM + TabPFN resources."""

    def __init__(self, resource_cache: dict, model_path: str, norm_path: str):
        self.resources = resource_cache

        # Load normalization stats
        norm = np.load(norm_path)
        self.X_mean = norm['X_mean']
        self.X_std = norm['X_std']
        self.y_mean = float(norm['y_mean'])
        self.y_std = float(norm['y_std'])

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BiLSTM(input_dim=17)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f'  Bi-LSTM loaded: {sum(p.numel() for p in self.model.parameters()):,} params')

    def _build_interaction(self, qi, si, peers):
        """
        Build 17-dim interaction vectors for target query qi.
        peers: [(start_time, qid), ...] sorted by start time.
        """
        if qi not in self.resources:
            return None, 0.5
        pred_i = self.resources[qi]
        serial_lat = pred_i.get('serial_lat_s', 10.0)
        qv = [pred_i[d] for d in DIMS]
        tr_ = [pred_i[d] for d in DIMS]

        seq = []
        for osv, oq in peers:
            if oq not in self.resources:
                continue
            pred_j = self.resources[oq]
            ovv = [pred_j[d] for d in DIMS]
            oc = [pred_j[d] for d in DIMS]
            c = self._resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        return seq, serial_lat

    @staticmethod
    def _resource_conflict(t, c):
        t_arr = np.array(t); c_arr = np.array(c)
        return list(np.minimum(t_arr, c_arr) /
                    np.maximum(np.abs(t_arr) + np.abs(c_arr) + 1e-8, 1e-8))

    def predict_system_runtime(self, target_qid: str,
                                concurrent_qids: List[str],
                                start_times: Dict[str, float] = None) -> float:
        """
        Predict system runtime (seconds) = serial_latency × slowdown_ratio.

        Builds 17-dim interaction vectors, sorted by start_time,
        feeds through Bi-LSTM to get slowdown ratio.
        """
        if start_times is None:
            start_times = {}

        # Build sorted peer list
        t_target = start_times.get(target_qid, 0.0)
        peers = [(start_times.get(cq, 0.0), cq) for cq in concurrent_qids
                 if cq != target_qid and cq in self.resources]
        peers.sort()

        seq, serial_lat = self._build_interaction(target_qid, t_target, peers)
        if not seq:
            return serial_lat * 1.5  # fallback

        # Stack + normalize + predict
        X = np.stack(seq, dtype=np.float32)  # [L, 17]
        X = (X - self.X_mean) / self.X_std
        X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
        L = torch.LongTensor([len(seq)]).to(self.device)

        with torch.no_grad():
            pred_z = self.model(X_t, L).cpu().item()

        ratio_pred = np.exp(pred_z * self.y_std + self.y_mean) - 1
        slowdown = max(ratio_pred, 0.01)
        return serial_lat * slowdown


# ═══════════ Query ═══════════

class Query:
    __slots__ = ('qid', 'arrival_time', 'serial_latency', 'start_time',
                 'finish_time', 'status', 'slowdown')
    def __init__(self, qid, arrival_time, serial_latency):
        self.qid = qid
        self.arrival_time = arrival_time
        self.serial_latency = serial_latency
        self.start_time = None
        self.finish_time = None
        self.status = 'waiting'
        self.slowdown = None


# ═══════════ ICONQ-Style Scheduler ═══════════

class TabpfnScheduler:
    """Event-driven greedy scheduler using ICONQ's algorithm."""

    def __init__(self, predictor: SlowdownPredictor,
                 max_concurrent: int = MAX_CONCURRENT,
                 theta: float = THETA):
        self.predictor = predictor
        self.max_concurrent = max_concurrent
        self.theta = theta
        self.running: Dict[str, Query] = {}
        self.waiting: deque = deque()
        self.completed: List[Query] = []
        self.now = 0.0

    def _score_query(self, wq: Query) -> float:
        rq_qids = list(self.running.keys())
        st = {rq.qid: rq.start_time for rq in self.running.values()
              if rq.start_time is not None}
        st[wq.qid] = self.now

        # Part A: system runtime of WQi in new concurrent set
        rt_wq = self.predictor.predict_system_runtime(
            wq.qid, rq_qids + [wq.qid], st)

        # Part B: delta slowdown inflicted on each running query
        total_delta = 0.0
        for rq in self.running.values():
            rt_old = self.predictor.predict_system_runtime(rq.qid, rq_qids, st)
            rt_new = self.predictor.predict_system_runtime(
                rq.qid, rq_qids + [wq.qid], st)
            total_delta += max(rt_new - rt_old, 0.0)

        return rt_wq + total_delta

    def _is_beneficial(self, wq: Query, score: float) -> bool:
        return score < wq.serial_latency * self.theta

    def _try_submit(self) -> bool:
        if not self.waiting or len(self.running) >= self.max_concurrent:
            return False
        best_score = float('inf'); best_query = None
        for wq in list(self.waiting):
            s = self._score_query(wq)
            if s < best_score:
                best_score = s; best_query = wq
        if best_query is None or not self._is_beneficial(best_query, best_score):
            return False
        # Submit
        self.waiting.remove(best_query)
        best_query.start_time = self.now
        best_query.status = 'running'
        self.running[best_query.qid] = best_query
        sys_rt = self.predictor.predict_system_runtime(
            best_query.qid, list(self.running.keys()),
            {rq.qid: rq.start_time for rq in self.running.values()
             if rq.start_time is not None})
        best_query.slowdown = sys_rt / best_query.serial_latency
        best_query.finish_time = self.now + sys_rt
        return True

    def _complete_query(self, rq: Query):
        rq.status = 'completed'
        del self.running[rq.qid]
        self.completed.append(rq)

    def _on_arrival(self, q: Query):
        self.waiting.append(q)

    def _next_completion(self) -> Optional[float]:
        if not self.running:
            return None
        return min(rq.finish_time for rq in self.running.values()
                   if rq.finish_time is not None)

    def run(self, arrivals: List[Tuple[str, float]]) -> List[Query]:
        arrivals = sorted(arrivals, key=lambda x: x[1])
        n_total = len(arrivals); arr_idx = 0

        query_pool = {}
        for qid, arr_t in arrivals:
            entry = self.predictor.resources.get(qid)
            if entry is None: continue
            query_pool[qid] = Query(qid, arr_t, entry.get('serial_lat_s', 10.0))
        print(f'  Queries: {len(query_pool)} / {n_total}')

        step = 0
        while arr_idx < n_total or self.waiting or self.running:
            step += 1
            if step > 200000: print('  WARNING: max steps'); break
            next_arr = (arrivals[arr_idx][1] if arr_idx < n_total else float('inf'))
            next_comp = self._next_completion() or float('inf')
            next_event = min(next_arr, next_comp)
            if next_event == float('inf'): break
            if next_event > self.now: self.now = next_event

            while arr_idx < n_total:
                qid, arr_t = arrivals[arr_idx]
                if arr_t <= self.now + 0.001:
                    if qid in query_pool: self._on_arrival(query_pool[qid])
                    arr_idx += 1
                else: break

            done = [rq for rq in self.running.values()
                    if rq.finish_time and rq.finish_time <= self.now + 0.001]
            for rq in done: self._complete_query(rq)

            submitted = True
            while submitted: submitted = self._try_submit()

        for rq in list(self.running.values()): self._complete_query(rq)
        return self.completed


# ═══════════ Output ═══════════

def write_trace(completed: List[Query], out_path: str):
    completed.sort(key=lambda q: q.arrival_time)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'qid', 'arrival', 'start', 'finish', 'runtime', 'status'])
        w.writeheader()
        for q in completed:
            w.writerow({
                'qid': q.qid,
                'arrival': round(q.arrival_time, 1),
                'start': round(q.start_time, 1) if q.start_time else '',
                'finish': round(q.finish_time, 1) if q.finish_time else '',
                'runtime': (round(q.finish_time - q.start_time, 2)
                            if q.start_time and q.finish_time else ''),
                'status': q.status,
            })
    print(f'  Trace → {out_path} ({len(completed)} queries)')


def print_stats(completed: List[Query]):
    e2e = []
    for q in completed:
        if q.finish_time: e2e.append(q.finish_time - q.arrival_time)
    if not e2e: return
    e2e.sort(); n = len(e2e)
    print(f'\n  E2E latency (n={n}):')
    print(f'    Mean={np.mean(e2e):.0f}s  P50={e2e[n//2]:.0f}s  '
          f'P90={e2e[int(n*0.9)]:.0f}s  P95={e2e[int(n*0.95)]:.0f}s')


# ═══════════ Main ═══════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trace', default='trace_2_mixed')
    p.add_argument('--max-concurrent', type=int, default=MAX_CONCURRENT)
    p.add_argument('--theta', type=float, default=THETA)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    print('=' * 60)
    print('TabPFN Scheduler (ICONQ greedy algorithm)')
    print(f'  max_concurrent={args.max_concurrent}, theta={args.theta}')
    print('=' * 60)

    # Load resource cache
    if not os.path.exists(RESOURCE_CACHE):
        print(f'ERROR: Resource cache not found: {RESOURCE_CACHE}')
        return
    with open(RESOURCE_CACHE) as f:
        resources = json.load(f)
    print(f'\n[1] Resources: {len(resources)} queries')

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model not found: {MODEL_PATH}')
        print('Run lstm/train_bilstm_tabpfn.py first.')
        return
    predictor = SlowdownPredictor(resources, MODEL_PATH, NORM_PATH)

    # Load trace
    trace_path = os.path.join(ROOT, 'collect_concurrent',
                               f'{args.trace}.csv')
    if not os.path.exists(trace_path):
        print(f'ERROR: {trace_path} not found')
        return
    arrivals = []
    with open(trace_path) as f:
        for row in csv.DictReader(f):
            arrivals.append((row['qid'], float(row['start'])))
    print(f'\n[2] Trace: {len(arrivals)} arrivals, '
          f'span={arrivals[-1][1]-arrivals[0][1]:.0f}s')

    # Filter to queries that have resource predictions
    has_pred = sum(1 for qid, _ in arrivals if qid in resources)
    print(f'  Queries with predictions: {has_pred}/{len(arrivals)}')

    # Run
    import time as _time
    print(f'\n[3] Running...')
    t0 = _time.time()
    scheduler = TabpfnScheduler(predictor=predictor,
                                 max_concurrent=args.max_concurrent,
                                 theta=args.theta)
    completed = scheduler.run(arrivals)
    print(f'  Done in {_time.time()-t0:.1f}s')

    out = args.output or os.path.join(
        ROOT, 'collect_concurrent',
        f'tabpfn_{args.trace}.csv')
    write_trace(completed, out)
    print_stats(completed)
    print('\nDone.')


if __name__ == '__main__':
    main()
