"""
ICONQ-Style Scheduler — replicates the scheduling algorithm from:
  "Improving DBMS Scheduling Decisions with Fine-grained
   Performance Prediction on Concurrent Queries" (VLDB 2025, MIT + AWS)

Key design decisions (matching the paper):
  - Event-driven: decisions only at query ARRIVAL or COMPLETION
  - Greedy scoring: for each waiting query, compute:
      score = sys_runtime(WQi | RQ + WQi) + Σ delta_sys_runtime(RQj)
    Lower score = better to submit NOW (least harm + fastest self-execution)
  - System runtime predictor: trained Bi-LSTM (ICONQ-style 97-dim features)
    * Replaces the paper's Bi-LSTM with our trained IconqBiLSTM from train_bilstm_iconq.py

Feature encoding (ICONQ Section 3.1):
  Query feature vector (47 dims per query):
    - 19 operator types × 2 stats (count, log(1+estRows)) = 38 dims
    - 8 tables × log(1+max_estRows)                    =  8 dims
    - XGBoost predicted serial latency (Stage proxy)    =  1 dim
  Interaction feature vector (97 dims):
    - Qi's 47 query features + Qj's (target) 47 query features
    - 3 timestamp features: |ti-tj|, 1(ti<tj), 1(tj<ti)

Usage:
  # Step 1: Train the Bi-LSTM
  python lstm/train_bilstm_iconq.py
  (Saves model + stats to checkpoints/iconq_bilstm.pt, checkpoints/iconq_norm.npz)

  # Step 2: Pre-compute ICONQ features
  python collect_concurrent/iconq_scheduler.py --cache-features

  # Step 3: Run scheduler (simulation mode)
  python collect_concurrent/iconq_scheduler.py --trace trace_2_mixed
"""
import os, sys, re, math, json, csv, time as _time, numpy as np
import torch, torch.nn as nn
from collections import deque
from typing import Dict, List, Tuple, Optional

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
FEATURE_CACHE = os.path.join(CKPT_DIR, 'iconq_query_features.json')
MODEL_PATH = os.path.join(CKPT_DIR, 'iconq_bilstm.pt')
NORM_PATH = os.path.join(CKPT_DIR, 'iconq_norm.npz')

# ═══ ICONQ Feature Encoding ═══

OP_TYPES = [
    'TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
    'IndexLookUp', 'IndexReader',
    'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
    'HashAgg', 'StreamAgg',
    'Sort', 'TopN', 'Window',
    'ExchangeSender', 'ExchangeReceiver',
    'Projection', 'Selection',
]
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer',
          'nation', 'region']


def extract_iconq_query_features(qid: str) -> Optional[np.ndarray]:
    """Extract 46-dim ICONQ plan features (runtime added separately for 47)."""
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        return None
    with open(pf) as f:
        plan = f.read()
    oc = {o: 0 for o in OP_TYPES}
    oe = {o: 0.0 for o in OP_TYPES}
    tc = {t: 0.0 for t in TABLES}
    for line in plan.split('\n'):
        if '\t' not in line or line.startswith('--'):
            continue
        parts = line.lstrip(' │├└─').split('\t')
        if len(parts) < 5:
            continue
        op = re.sub(r'^[│├└─\s]+', '', parts[0].strip())
        op = re.sub(r'\(Build\)|\(Probe\)', '', op).strip()
        op = re.sub(r'_\d+$', '', op)
        try:
            est = float(parts[1].strip())
        except ValueError:
            est = 1.0
        if op in oc:
            oc[op] += 1
            oe[op] += est
        oi = parts[4].strip() if len(parts) > 4 else ''
        for t in TABLES:
            if t in oi.lower():
                tc[t] = max(tc[t], est)
    feat = []
    for o in OP_TYPES:
        feat.append(float(oc[o]))
        feat.append(math.log(1 + oe[o]))
    for t in TABLES:
        feat.append(math.log(1 + tc[t]))
    return np.array(feat, dtype=np.float32)


def make_query_vector(qid: str, features: dict) -> Optional[np.ndarray]:
    """Build 47-dim query vector: 46 plan + 1 runtime."""
    entry = features.get(qid)
    if entry is None:
        return None
    plan = np.array(entry['plan_46d'], dtype=np.float32)
    runtime = np.array([entry['xgb_lat_log']], dtype=np.float32)
    return np.concatenate([plan, runtime])


def make_interaction_vector(qi: np.ndarray, qj: np.ndarray,
                             ti: float, tj: float) -> np.ndarray:
    """97-dim interaction vector: Qi(47) + Qj(47) + 3 timestamp."""
    return np.concatenate([
        qi, qj,
        np.array([abs(ti - tj),
                  1.0 if ti < tj else 0.0,
                  1.0 if tj < ti else 0.0], dtype=np.float32),
    ])


# ═══ Feature Cache ═══

def cache_all_features():
    print('=' * 60)
    print('Caching ICONQ query features...')
    print('=' * 60)
    xgb_path = os.path.join(CKPT_DIR, 'oof_xgboost_k5.json')
    with open(xgb_path) as f:
        xgb_cache = json.load(f)
    print(f'  XGBoost latency cache: {len(xgb_cache)} queries')

    sys.path.insert(0, os.path.join(ROOT, 'gnn'))
    from train_cgroup import load_cgroup_labels
    cgroup = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    cgroup_qids = set(cgroup.keys())

    all_qids = sorted(set(xgb_cache.keys()) & cgroup_qids)
    cache = {}
    n_ok = 0
    for qid in all_qids:
        plan_feat = extract_iconq_query_features(qid)
        if plan_feat is None:
            continue
        lat_log = xgb_cache[qid].get('lat')
        if lat_log is None:
            continue
        cache[qid] = {
            'plan_46d': plan_feat.tolist(),
            'xgb_lat_log': lat_log,
            'xgb_lat_s': max(math.exp(lat_log) - 1, 0.5),
        }
        n_ok += 1
    with open(FEATURE_CACHE, 'w') as f:
        json.dump(cache, f)
    print(f'  Cached: {n_ok} queries → {FEATURE_CACHE}')
    print('Done.')


# ═══ Bi-LSTM Model (same as train_bilstm_iconq.py) ═══

class IconqBiLSTM(nn.Module):
    def __init__(self, input_dim=97, embed_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                              dropout=dropout, batch_first=True,
                              bidirectional=True)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, X, lengths):
        if X.shape[1] > 1:
            X = torch.transpose(X, 1, 2)
            X = self.bn(X)
            X = torch.transpose(X, 1, 2)
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[torch.arange(len(lengths)), lengths - 1]
        return self.output_layer(output).squeeze(-1)


# ═══ System Runtime Predictor (Bi-LSTM powered) ═══

class SystemRuntimePredictor:
    """ICONQ-style predictor using trained Bi-LSTM."""

    def __init__(self, features: dict, model_path: str, norm_path: str):
        self.features = features

        # Load normalization stats
        norm = np.load(norm_path)
        self.X_mean = norm['X_mean']
        self.X_std = norm['X_std']
        self.y_mean = float(norm['y_mean'])
        self.y_std = float(norm['y_std'])

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IconqBiLSTM(input_dim=97)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f'  Bi-LSTM loaded: {sum(p.numel() for p in self.model.parameters()):,} params')

    def predict_system_runtime(self, target_qid: str,
                                concurrent_qids: List[str],
                                start_times: Dict[str, float] = None) -> float:
        """
        Predict system runtime (seconds) of target_qid running with concurrent_qids.

        start_times: {qid: actual_start_time_s} for each query.
                     target_qid's start_time = when it would begin (now).
                     Concurrent queries' start_times = when scheduler submitted them.
        """
        if start_times is None:
            start_times = {}

        all_qids = concurrent_qids + [target_qid]
        all_qids = list(dict.fromkeys(all_qids))  # dedup, preserve order

        q_vectors = {}
        for qid in all_qids:
            qv = make_query_vector(qid, self.features)
            if qv is not None:
                q_vectors[qid] = qv

        if target_qid not in q_vectors:
            return self.features.get(target_qid, {}).get('xgb_lat_s', 10.0) * 2.0

        target_vec = q_vectors[target_qid]
        t_target = start_times.get(target_qid, 0.0)

        # Build interaction vectors sorted by start_time (ICONQ: order by submission time)
        entries = []
        for qid in all_qids:
            if qid not in q_vectors:
                continue
            t_i = start_times.get(qid, 0.0)
            entries.append((t_i, qid))
        entries.sort()

        interaction_vecs = []
        for t_i, qid in entries:
            interaction_vecs.append(
                make_interaction_vector(q_vectors[qid], target_vec, t_i, t_target))

        if not interaction_vecs:
            return self.features.get(target_qid, {}).get('xgb_lat_s', 10.0) * 2.0

        # Stack into batch [1, L, 97]
        X = np.stack(interaction_vecs)  # [L, 97]
        X = (X - self.X_mean) / self.X_std
        X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # [1, L, 97]
        L = torch.LongTensor([len(interaction_vecs)]).to(self.device)

        with torch.no_grad():
            pred_z = self.model(X_t, L).cpu().item()

        # Denormalize: z-score → log(1+seconds) → seconds
        pred_log = pred_z * self.y_std + self.y_mean
        return max(math.exp(pred_log) - 1, 0.01)


# ═══ Query Data Structure ═══

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

    def __repr__(self):
        return (f"Q({self.qid}, arr={self.arrival_time:.0f}s, "
                f"ser={self.serial_latency:.1f}s, {self.status})")


# ═══ ICONQ Scheduler ═══

class IconqScheduler:
    def __init__(self, predictor: SystemRuntimePredictor,
                 max_concurrent: int = 10, theta: float = 1.5):
        self.predictor = predictor
        self.max_concurrent = max_concurrent
        self.theta = theta
        self.running: Dict[str, Query] = {}
        self.waiting: deque = deque()
        self.completed: List[Query] = []
        self.now = 0.0
        self.event_log: List[Tuple[float, str, str]] = []

    def _score_query(self, wq: Query) -> float:
        rq_qids = list(self.running.keys())
        # start_times: running queries' actual start + candidate WQi at now
        st = {rq.qid: rq.start_time for rq in self.running.values()
              if rq.start_time is not None}
        st[wq.qid] = self.now

        rt_wq = self.predictor.predict_system_runtime(wq.qid, rq_qids + [wq.qid], st)
        total_delta = 0.0
        for rq in self.running.values():
            rt_old = self.predictor.predict_system_runtime(rq.qid, rq_qids, st)
            rt_new = self.predictor.predict_system_runtime(rq.qid, rq_qids + [wq.qid], st)
            total_delta += max(rt_new - rt_old, 0.0)
        return rt_wq + total_delta

    def _is_beneficial(self, wq: Query, score: float) -> bool:
        return score < wq.serial_latency * self.theta

    def _try_submit(self) -> bool:
        if not self.waiting or len(self.running) >= self.max_concurrent:
            return False
        best_score = float('inf')
        best_query = None
        for wq in list(self.waiting):
            s = self._score_query(wq)
            if s < best_score:
                best_score = s
                best_query = wq
        if best_query is None or not self._is_beneficial(best_query, best_score):
            return False
        self.waiting.remove(best_query)
        best_query.start_time = self.now
        best_query.status = 'running'
        self.running[best_query.qid] = best_query
        self.event_log.append((self.now, best_query.qid, 'start'))
        sys_rt = self.predictor.predict_system_runtime(
            best_query.qid, list(self.running.keys()),
            {rq.qid: rq.start_time for rq in self.running.values()
             if rq.start_time is not None})
        best_query.slowdown = sys_rt / best_query.serial_latency
        best_query.finish_time = self.now + sys_rt
        return True

    def _complete_query(self, rq: Query):
        rq.status = 'completed'
        self.event_log.append((self.now, rq.qid, 'finish'))
        del self.running[rq.qid]
        self.completed.append(rq)

    def _on_arrival(self, q: Query):
        self.event_log.append((self.now, q.qid, 'arrival'))
        self.waiting.append(q)

    def _next_completion(self) -> Optional[float]:
        if not self.running:
            return None
        return min(rq.finish_time for rq in self.running.values()
                   if rq.finish_time is not None)

    def run(self, arrivals: List[Tuple[str, float]]) -> List[Query]:
        arrivals = sorted(arrivals, key=lambda x: x[1])
        n_total = len(arrivals)
        arr_idx = 0
        query_pool = {}
        for qid, arr_t in arrivals:
            entry = self.predictor.features.get(qid)
            if entry is None:
                continue
            query_pool[qid] = Query(qid, arr_t, entry['xgb_lat_s'])
        print(f'  Queries: {len(query_pool)} / {n_total}')

        step = 0
        while arr_idx < n_total or self.waiting or self.running:
            step += 1
            if step > 200000:
                print('  WARNING: max steps')
                break
            next_arr = (arrivals[arr_idx][1] if arr_idx < n_total else float('inf'))
            next_comp = self._next_completion() or float('inf')
            next_event = min(next_arr, next_comp)
            if next_event == float('inf'):
                break
            if next_event > self.now:
                self.now = next_event
            while arr_idx < n_total:
                qid, arr_t = arrivals[arr_idx]
                if arr_t <= self.now + 0.001:
                    if qid in query_pool:
                        self._on_arrival(query_pool[qid])
                    arr_idx += 1
                else:
                    break
            done = [rq for rq in self.running.values()
                    if rq.finish_time is not None
                    and rq.finish_time <= self.now + 0.001]
            for rq in done:
                self._complete_query(rq)
            submitted = True
            while submitted:
                submitted = self._try_submit()
        for rq in list(self.running.values()):
            self._complete_query(rq)
        return self.completed


# ═══ Output ═══

def write_trace(completed: List[Query], out_path: str):
    completed.sort(key=lambda q: q.arrival_time)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'qid', 'arrival', 'start', 'finish', 'runtime', 'status'])
        w.writeheader()
        for q in completed:
            w.writerow({
                'qid': q.qid, 'arrival': round(q.arrival_time, 1),
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
        if q.finish_time:
            e2e.append(q.finish_time - q.arrival_time)
    if not e2e:
        return
    e2e.sort()
    n = len(e2e)
    print(f'\n  E2E latency (n={n}):')
    print(f'    Mean={np.mean(e2e):.0f}s  P50={e2e[n//2]:.0f}s  '
          f'P90={e2e[int(n*0.9)]:.0f}s  P95={e2e[int(n*0.95)]:.0f}s')


# ═══ Main ═══

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cache-features', action='store_true')
    p.add_argument('--trace', default='trace_2_mixed')
    p.add_argument('--max-concurrent', type=int, default=10)
    p.add_argument('--theta', type=float, default=1.5)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.cache_features:
        cache_all_features()
        return

    if not os.path.exists(FEATURE_CACHE):
        print(f'Feature cache not found: {FEATURE_CACHE}')
        print('Run with --cache-features first.')
        return
    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}')
        print('Run train_bilstm_iconq.py first.')
        return

    print('=' * 60)
    print('ICONQ Scheduler (Bi-LSTM predictor)')
    print(f'  max_concurrent={args.max_concurrent}, theta={args.theta}')
    print('=' * 60)

    with open(FEATURE_CACHE) as f:
        features = json.load(f)
    print(f'\n[1] Features: {len(features)} queries')

    predictor = SystemRuntimePredictor(features, MODEL_PATH, NORM_PATH)

    trace_path = os.path.join(ROOT, 'collect_concurrent',
                               f'{args.trace}.csv')
    arrivals = []
    with open(trace_path) as f:
        for row in csv.DictReader(f):
            arrivals.append((row['qid'], float(row['start'])))
    print(f'\n[2] Trace: {len(arrivals)} arrivals, '
          f'span={arrivals[-1][1]-arrivals[0][1]:.0f}s')

    print(f'\n[3] Running...')
    t0 = _time.time()
    scheduler = IconqScheduler(predictor=predictor,
                                max_concurrent=args.max_concurrent,
                                theta=args.theta)
    completed = scheduler.run(arrivals)
    print(f'  Done in {_time.time()-t0:.1f}s')

    out = args.output or os.path.join(
        ROOT, 'collect_concurrent',
        f'iconq_{args.trace}.csv')
    write_trace(completed, out)
    print_stats(completed)
    print('\nDone.')


if __name__ == '__main__':
    main()
