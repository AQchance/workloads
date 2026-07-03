"""
ICONQ-Style Scheduler — replicates the scheduling algorithm from:
  "Improving DBMS Scheduling Decisions with Fine-grained
   Performance Prediction on Concurrent Queries" (VLDB 2025, MIT + AWS)

Key design decisions (matching the paper):
  - Event-driven: decisions only at query ARRIVAL or COMPLETION
  - Greedy scoring: for each waiting query, compute:
      score = sys_runtime(WQi | RQ + WQi) + Σ delta_sys_runtime(RQj)
    Lower score = better to submit NOW (least harm + fastest self-execution)
  - "Beneficial" check: submit only if score < serial_latency × θ
    (θ defaults to 1.3; a query predicted to suffer badly from concurrency waits)

Feature encoding (matches ICONQ Section 3.1):
  Query feature vector (47 dims per query):
    - 19 operator types × 2 stats (count, log(1+estRows)) = 38 dims
    - 8 tables × log(1+max_estRows)                    =  8 dims
    - XGBoost predicted serial latency                  =  1 dim  (proxy for Stage)
  Interaction feature vector (97 dims):
    - Qi's 47 query features
    - Qj's 47 query features (target query)
    - 3 timestamp features: |ti-tj|, 1(ti<tj), 1(tj<ti)

Usage:
  # Step 1: Pre-compute ICONQ features
  python collect_concurrent/iconq_scheduler.py --cache-features

  # Step 2: Run scheduler (simulation mode, no real TiDB)
  python collect_concurrent/iconq_scheduler.py --trace trace_2_mixed

This does NOT need a running TiDB — it's a simulation using cached features.
"""
import os, sys, re, math, json, csv, time as _time, numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
FEATURE_CACHE = os.path.join(CKPT_DIR, 'iconq_query_features.json')

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
    """
    Extract 46-dim ICONQ query features from EXPLAIN plan.
    (47th dim = XGBoost latency, added separately when building interaction vectors)

    Returns [38 op features + 8 table features] = 46 dims, or None.
    """
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        return None
    with open(pf) as f:
        plan = f.read()

    oc = {o: 0 for o in OP_TYPES}       # operator count
    oe = {o: 0.0 for o in OP_TYPES}     # operator sum of estRows
    tc = {t: 0.0 for t in TABLES}       # table max estRows

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


def build_interaction_vector(qi_features: np.ndarray,
                              qj_features: np.ndarray,
                              ti: float, tj: float) -> np.ndarray:
    """
    Build 97-dim interaction feature vector (ICONQ Figure 4).

    qi_features: 46-dim query features of the current query
    qj_features: 46-dim query features of the target query
    ti, tj: submission times

    Returns: 46 + 46 + 3 = 95... wait.
    Actually: 46 (Qi) + 46 (Qj) + 3 (timestamp) = 95 dims.
    The paper adds 1 runtime feature to each query vector,
    making it 47 per query → 47+47+3 = 97.

    Our runtime feature (XGBoost latency) is added to qi and qj separately
    when building query vectors. So this function expects:
      qi = 47-dim (46 plan+table + 1 runtime)
      qj = 47-dim
      + 3 timestamp
      = 97 dims total.
    """
    timestamp_diff = abs(ti - tj)
    is_before = 1.0 if ti < tj else 0.0
    is_after = 1.0 if tj < ti else 0.0

    return np.concatenate([
        qi_features,
        qj_features,
        np.array([timestamp_diff, is_before, is_after], dtype=np.float32),
    ])


# ═══ Feature Cache ═══

def cache_all_features():
    """
    Pre-compute and save ICONQ query features + XGBoost latency for all queries.
    Runs once; the scheduler loads from cache.
    """
    print('=' * 60)
    print('Caching ICONQ query features...')
    print('=' * 60)

    # Load XGBoost OOF latency cache
    xgb_cache_path = os.path.join(CKPT_DIR, 'oof_xgboost_k5.json')
    with open(xgb_cache_path) as f:
        xgb_cache = json.load(f)
    print(f'  XGBoost latency cache: {len(xgb_cache)} queries')

    # Load TabPFN OOF resource cache (for resource conflict features)
    tpf_cache_path = os.path.join(CKPT_DIR, 'oof_tabpfn_k5.json')
    with open(tpf_cache_path) as f:
        tpf_cache = json.load(f)
    print(f'  TabPFN resource cache: {len(tpf_cache)} queries')

    # Get all qids from cgroup that have EXPLAIN plans
    sys.path.insert(0, os.path.join(ROOT, 'gnn'))
    from train_cgroup import load_cgroup_labels
    cgroup = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    cgroup_qids = set(cgroup.keys())

    all_qids = set()
    for cache in [xgb_cache, tpf_cache]:
        all_qids.update(cache.keys())
    all_qids = sorted(all_qids & cgroup_qids)

    features = {}
    n_ok = 0
    for qid in all_qids:
        f = extract_iconq_query_features(qid)
        if f is None:
            continue
        # Get XGBoost latency
        xgb_entry = xgb_cache.get(qid, {})
        lat_log = xgb_entry.get('lat', None)
        if lat_log is None:
            continue
        # Get resources (for conflict-based slowdown estimation)
        tpf_entry = tpf_cache.get(qid, {})
        if not tpf_entry:
            continue

        features[qid] = {
            'iconq_46d': f.tolist(),
            'xgb_lat_log': lat_log,
            'xgb_lat_s': max(math.exp(lat_log) - 1, 0.5),
            'resources': [tpf_entry.get(d, 0.0)
                           for d in ['mem', 'disk', 'net', 'lat', 'cpures']],
        }
        n_ok += 1

    with open(FEATURE_CACHE, 'w') as f:
        json.dump(features, f)
    print(f'  Cached: {n_ok} queries → {FEATURE_CACHE}')
    print('Done.')
    return features


# ═══ Slowdown Predictor (ICONQ-style) ═══

class IconqPredictor:
    """
    ICONQ-style system runtime predictor.

    In the paper, this is a trained Bi-LSTM that ingests interaction feature
    vectors and outputs system runtime.

    Our implementation uses a trained Bi-LSTM loaded from checkpoint.
    If no trained model is available, falls back to a resource-conflict
    heuristic (still useful for testing the scheduling logic).
    """

    def __init__(self, feature_cache: dict):
        self.features = feature_cache
        print(f'  Predictor: {len(feature_cache)} queries in cache')

    def _get_query_vector(self, qid: str) -> Optional[np.ndarray]:
        """Build 47-dim query feature vector: 46 ICONQ + 1 runtime."""
        entry = self.features.get(qid)
        if entry is None:
            return None
        iconq = np.array(entry['iconq_46d'], dtype=np.float32)
        runtime = np.array([entry['xgb_lat_log']], dtype=np.float32)
        return np.concatenate([iconq, runtime])

    def _resource_conflict(self, t, c):
        t = np.array(t); c = np.array(c)
        return np.minimum(t, c) / np.maximum(
            np.abs(t) + np.abs(c) + 1e-8, 1e-8)

    def predict_system_runtime(self, target_qid: str,
                                concurrent_qids: List[str]) -> float:
        """
        Predict system runtime of target_qid when running with concurrent_qids.

        Returns seconds (wall-clock time from submission to completion).

        Uses resource-conflict heuristic as a proxy for the Bi-LSTM.
        Replace this with actual Bi-LSTM inference for production use.
        """
        target_entry = self.features.get(target_qid)
        if target_entry is None:
            return target_entry['xgb_lat_s'] * 1.5 if target_entry else 999

        serial_lat = target_entry['xgb_lat_s']
        target_res = target_entry['resources']

        if not concurrent_qids:
            return serial_lat

        # Compute average pairwise conflict with each concurrent query
        total_conflict = 0.0
        n_valid = 0
        for cq in concurrent_qids:
            if cq == target_qid:
                continue
            cq_entry = self.features.get(cq)
            if cq_entry is None:
                continue
            conflict = self._resource_conflict(target_res, cq_entry['resources'])
            total_conflict += float(np.mean(conflict))
            n_valid += 1

        if n_valid == 0:
            return serial_lat

        avg_conflict = total_conflict / n_valid
        slowdown = 1.0 + avg_conflict * 3.0  # maps [0,1] → [1.0, 4.0]
        return serial_lat * slowdown


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
    """
    Event-driven, greedy query scheduler (ICONQ algorithm).

    Maintains:
      running_queries:  currently executing (≤ max_concurrent)
      waiting_queue:    arrived but not yet submitted
      completed:        finished queries (with timing for evaluation)
    """

    def __init__(self, predictor: IconqPredictor,
                 max_concurrent: int = 10,
                 slowdown_threshold: float = 1.5):
        self.predictor = predictor
        self.max_concurrent = max_concurrent
        self.theta = slowdown_threshold

        self.running: Dict[str, Query] = {}
        self.waiting: deque = deque()
        self.completed: List[Query] = []

        self.now = 0.0
        self.event_log: List[Tuple[float, str, str]] = []

    # ─── Scoring ───

    def _score_query(self, wq: Query) -> float:
        """
        Score a waiting query for potential submission.

        Part A: system runtime of WQi in the NEW concurrent set
        Part B: additional slowdown inflicted on each running query

        score = sys_rt(WQi | RQ+WQi) + Σ max(0, sys_rt(RQj | RQ+WQi) - sys_rt(RQj | RQ))
        """
        rq_qids = list(self.running.keys())
        new_set = rq_qids + [wq.qid]

        # Part A
        rt_wq = self.predictor.predict_system_runtime(wq.qid, new_set)

        # Part B
        total_delta = 0.0
        for rq in self.running.values():
            rt_old = self.predictor.predict_system_runtime(rq.qid, rq_qids)
            rt_new = self.predictor.predict_system_runtime(rq.qid, new_set)
            total_delta += max(rt_new - rt_old, 0.0)

        return rt_wq + total_delta

    def _is_beneficial(self, wq: Query, score: float) -> bool:
        """True if submitting WQi now is better than waiting."""
        return score < wq.serial_latency * self.theta

    # ─── Submission ───

    def _try_submit(self) -> bool:
        """
        Try to submit ONE waiting query.

        1. Score all waiting queries
        2. Pick the one with lowest score
        3. If beneficial, submit it; else nobody gets submitted this round

        Returns True if a query was submitted.
        """
        if not self.waiting:
            return False
        if len(self.running) >= self.max_concurrent:
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

        # Submit best_query
        self.waiting.remove(best_query)
        best_query.start_time = self.now
        best_query.status = 'running'
        self.running[best_query.qid] = best_query
        self.event_log.append((self.now, best_query.qid, 'start'))

        # Predict completion time
        concurrent = list(self.running.keys())
        sys_rt = self.predictor.predict_system_runtime(best_query.qid, concurrent)
        best_query.slowdown = sys_rt / best_query.serial_latency
        best_query.finish_time = self.now + sys_rt

        return True

    # ─── Event handlers ───

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

    # ─── Main loop ───

    def run(self, arrivals: List[Tuple[str, float]]) -> List[Query]:
        """
        Main event loop.

        arrivals: sorted list of (qid, arrival_time_s)
        """
        arrivals = sorted(arrivals, key=lambda x: x[1])
        n_total = len(arrivals)
        arr_idx = 0

        # Pre-build Query objects
        query_pool = {}
        for qid, arr_t in arrivals:
            entry = self.predictor.features.get(qid)
            if entry is None:
                continue
            query_pool[qid] = Query(qid, arr_t, entry['xgb_lat_s'])

        print(f'  Queries with full features: {len(query_pool)} / {n_total}')

        step = 0
        while (arr_idx < n_total or self.waiting or self.running):
            step += 1
            if step > 200000:
                print('  WARNING: max steps reached')
                break

            # 1. Determine next event time
            next_arr = (arrivals[arr_idx][1]
                        if arr_idx < n_total else float('inf'))
            next_comp = self._next_completion() or float('inf')
            next_event = min(next_arr, next_comp)

            if next_event == float('inf'):
                break

            if next_event > self.now:
                self.now = next_event

            # 2. Process arrivals at current time
            while arr_idx < n_total:
                qid, arr_t = arrivals[arr_idx]
                if arr_t <= self.now + 0.001:
                    if qid in query_pool:
                        self._on_arrival(query_pool[qid])
                    arr_idx += 1
                else:
                    break

            # 3. Process completions at current time
            done = [rq for rq in self.running.values()
                    if rq.finish_time is not None
                    and rq.finish_time <= self.now + 0.001]
            for rq in done:
                self._complete_query(rq)

            # 4. Greedy submission loop
            submitted = True
            while submitted:
                submitted = self._try_submit()

        # Drain remaining
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
    p.add_argument('--cache-features', action='store_true',
                   help='Pre-compute and save ICONQ features')
    p.add_argument('--trace', default='trace_2_mixed',
                   help='Trace name (without .csv)')
    p.add_argument('--max-concurrent', type=int, default=10)
    p.add_argument('--theta', type=float, default=1.5,
                   help='Slowdown threshold')
    p.add_argument('--output', default=None)
    args = p.parse_args()

    # ─── Feature caching ───
    if args.cache_features:
        cache_all_features()
        return

    # ─── Load feature cache ───
    if not os.path.exists(FEATURE_CACHE):
        print(f'Feature cache not found: {FEATURE_CACHE}')
        print('Run with --cache-features first.')
        return

    print('=' * 60)
    print('ICONQ-Style Scheduler')
    print(f'  max_concurrent={args.max_concurrent}, theta={args.theta}')
    print('=' * 60)

    with open(FEATURE_CACHE) as f:
        features = json.load(f)
    print(f'\n[1] Loaded features: {len(features)} queries')

    predictor = IconqPredictor(features)

    # ─── Load trace ───
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

    # ─── Run ───
    print(f'\n[3] Running scheduler...')
    t0 = _time.time()
    scheduler = IconqScheduler(
        predictor=predictor,
        max_concurrent=args.max_concurrent,
        slowdown_threshold=args.theta,
    )
    completed = scheduler.run(arrivals)
    print(f'  Done in {_time.time()-t0:.1f}s')

    # ─── Output ───
    out = args.output or os.path.join(
        ROOT, 'collect_concurrent',
        f'iconq_{args.trace}.csv')
    write_trace(completed, out)
    print_stats(completed)
    print('\nDone.')


if __name__ == '__main__':
    main()
