"""
TabPFN resources → Bi-LSTM concurrent slowdown prediction.
Compares: TabPFN vs XGBoost as first-stage resource predictors for Bi-LSTM.

Two-stage pipeline:
  Stage 1: TabPFN (or XGBoost) predicts 5 resource dims for each query
  Stage 2: Bi-LSTM uses those resources to predict concurrent slowdown ratios

Same 80/10/10 query-level split and Bi-LSTM architecture as xgboost_bilstm.py.
"""
import os, sys, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = '/home/anqian/Desktop/my_lab/workloads'
sys.path.insert(0, os.path.join(ROOT, 'gnn'))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, load_dist_cache, parse_plan
from tabpfn import TabPFNRegressor
from sklearn.ensemble import GradientBoostingRegressor

# ═══ Feature Extraction ═══

def extract_flat_features(qid, ndv_cache):
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf): return None
    with open(pf) as f: plan_text = f.read()
    g = parse_plan(plan_text, ndv_cache)
    if g is None or g.x.shape[0] == 0: return None
    x = g.x.numpy()
    feats = []
    for col in range(x.shape[1]):
        vals = x[:, col]
        feats.extend([np.mean(vals), np.max(vals), np.sum(vals), np.std(vals)])
    feats.append(x.shape[0])
    feats.append(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0)
    return np.array(feats, dtype=np.float64)


DIM_NAMES = ['mem', 'disk', 'net', 'lat', 'cpures']

# ═══ Stage 1: Train Resource Predictors ═══

def train_tabpfn_predictors(ndv_cache, seed=42):
    """Train TabPFN on ALL cgroup data, return predictor for 5 resources."""
    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')

    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

    X_all, y_all = [], []
    for qid, lab in cgroup_labels.items():
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: continue
        X_all.append(feat)
        y_all.append([
            math.log(1 + max(lab['memory_bytes'], 1)),
            math.log(1 + max(lab['disk_bytes'], 1)),
            math.log(1 + max(lab['network_bytes'], 1)),
            math.log(1 + max(lab['latency_s'], 0.001)),
            math.log(1 + max(lab['cpu_resource_s'], 0.001)),
        ])

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f'  TabPFN training: {len(X_all)} queries, {X_all.shape[1]} features')

    models = []
    for i, name in enumerate(DIM_NAMES):
        print(f'    Training TabPFN for {name}...')
        m = TabPFNRegressor(n_estimators=8, random_state=seed,
                            fit_mode='low_memory', device='cuda')
        m.fit(X_all.astype(np.float32), y_all[:, i].astype(np.float32))
        models.append(m)

    def predict(qid):
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: return None
        preds = {}
        for i, name in enumerate(DIM_NAMES):
            preds[name] = float(models[i].predict(
                feat.astype(np.float32).reshape(1, -1))[0])
        return preds

    return predict


def train_xgboost_predictors(ndv_cache, seed=42):
    """Same as xgboost_bilstm.py"""
    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')

    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

    X_all, y_all = [], []
    for qid, lab in cgroup_labels.items():
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: continue
        X_all.append(feat)
        y_all.append([
            math.log(1 + max(lab['memory_bytes'], 1)),
            math.log(1 + max(lab['disk_bytes'], 1)),
            math.log(1 + max(lab['network_bytes'], 1)),
            math.log(1 + max(lab['latency_s'], 0.001)),
            math.log(1 + max(lab['cpu_resource_s'], 0.001)),
        ])

    X_all = np.array(X_all); y_all = np.array(y_all)
    Xm = X_all.mean(axis=0); Xs = X_all.std(axis=0) + 1e-8
    X_norm = (X_all - Xm) / Xs

    print(f'  XGBoost training: {len(X_all)} queries, {X_all.shape[1]} features')
    models = []
    for i, name in enumerate(DIM_NAMES):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                       learning_rate=0.1, random_state=seed,
                                       subsample=0.8)
        m.fit(X_norm, y_all[:, i])
        models.append(m)

    def predict(qid):
        feat = extract_flat_features(qid, ndv_cache)
        if feat is None: return None
        feat_n = (feat - Xm) / Xs
        return {name: float(models[i].predict(feat_n.reshape(1, -1))[0])
                for i, name in enumerate(DIM_NAMES)}

    return predict


# ═══ Stage 2: Build Concurrent Sequences ═══

def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_sequences(data_list, resource_predict, qid_info):
    X, y_ratio = [], []
    n_miss = 0
    for qi, si, ei, rti, sti, ov in data_list:
        if sti == 'penalty': continue
        pred_i = resource_predict(qi)
        if pred_i is None: n_miss += 1; continue
        serial_lat = max(math.exp(pred_i['lat']) - 1, 0.5)
        qv = [pred_i['mem'], pred_i['disk'], pred_i['net'], pred_i['lat'],
              pred_i['cpures'], math.log(1 + serial_lat)]
        tr_ = [pred_i['mem'], pred_i['disk'], pred_i['net'], pred_i['lat'],
               pred_i['cpures']]
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov
                 if oq in qid_info and resource_predict(oq) is not None]
        peers.sort()
        for osv, oq in peers:
            pred_j = resource_predict(oq)
            oslv = pred_j['lat']
            ovv = [pred_j['mem'], pred_j['disk'], pred_j['net'], pred_j['lat'],
                   pred_j['cpures'], oslv]
            oc = [pred_j['mem'], pred_j['disk'], pred_j['net'], pred_j['lat'],
                  pred_j['cpures']]
            c = resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    if n_miss: print(f'  Skipped {n_miss} queries (no features)')
    return X, y_ratio


# ═══ Stage 3: Bi-LSTM ═══

class RatioDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]


def collate_fn(batch):
    X, l, y = zip(*batch)
    si = torch.argsort(torch.stack(l), descending=True)
    return torch.stack([X[i] for i in si]), torch.stack([l[i] for i in si]), \
           torch.stack([y[i] for i in si])


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
    def forward(self, X, lengths):
        x = self.emb(X)
        p = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                               enforce_sorted=True)
        _, (hn, _) = self.lstm(p)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


def pad_and_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq): Xa[i, :len(s)] = s
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    if Xm is None:
        mask = np.zeros_like(Xa)
        for i, s in enumerate(X_seq): mask[i, :len(s)] = 1.0
        Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        Xs = np.sqrt(((Xa - Xm)**2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    X_norm = (Xa - Xm) / Xs; y_norm = (yl - ym) / ys
    return X_norm, lens, y_norm, Xm, Xs, ym, ys


def train_bilstm(train_ds, val_ds, test_ds, ym, ys, input_dim, seed=42, epochs=250):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              collate_fn=collate_fn)

    model = BiLSTM(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                            eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            model.eval(); ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    X = X.to(device)
                    ap.append(model(X, l).cpu().numpy())
                    at.append(y.numpy())
            pz = np.concatenate(ap); tz = np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
            if epoch % 100 == 0:
                print(f'  E{epoch:3d} val={med:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state); model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader:
            X = X.to(device)
            ap.append(model(X, l).cpu().numpy())
            at.append(y.numpy())
    pz = np.concatenate(ap); tz = np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
    tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    return qe, n_params


# ═══ Main ═══

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--skip-tabpfn', action='store_true')
    parser.add_argument('--skip-xgboost', action='store_true')
    args = parser.parse_args()

    print('=' * 60)
    print('TabPFN vs XGBoost → Bi-LSTM: Concurrent Slowdown Prediction')
    print(f'Seed={args.seed}, Epochs={args.epochs}')
    print('=' * 60)

    print('\nLoading data...')
    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))
    dist_cache = load_dist_cache(os.path.join(ROOT, 'dist_cache.json'))

    # Load timeline
    TRACES = ['collect_concurrent/trace_2_mixed.csv',
              'collect_concurrent/trace_3_fixed_mixed.csv',
              'collect_concurrent/trace_4_fixed_mixed.csv']
    timeline = []
    for tf in TRACES:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime'])
                actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']),
                                 float(row['start']) + actual,
                                 row['qid'], actual,
                                 row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    # Load GNN features (for cross-reference)
    gnf = {}
    for fn in ['lstm/gnn_features_k2_fixed.json',
               'lstm/gnn_features_k3_fixed.json',
               'lstm/gnn_features_k4_fixed.json']:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))

    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))
    print(f'  Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # 80/10/10 query-level split
    np.random.seed(args.seed)
    np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8)
    n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr])
    val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'  Query split: {len(train_qids)} train / {len(val_qids)} val / '
          f'{len(test_qids)} test')

    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]

    all_results = {}

    # ═══ TabPFN Pipeline ═══
    if not args.skip_tabpfn:
        print('\n' + '=' * 60)
        print('PIPELINE A: TabPFN → Bi-LSTM')
        print('=' * 60)

        print('\n[Stage 1] Training TabPFN resource predictors...')
        tpf_predict = train_tabpfn_predictors(ndv_cache, seed=args.seed)

        tpf_cache = {}
        for qid in unique_qids:
            pred = tpf_predict(qid)
            if pred: tpf_cache[qid] = pred
        print(f'  TabPFN predictions cached: {len(tpf_cache)} queries')

        def tpf_cached(qid): return tpf_cache.get(qid)

        X_tr, y_tr = build_sequences(train_d, tpf_cached, qid_info)
        X_va, y_va = build_sequences(val_d, tpf_cached, qid_info)
        X_te, y_te = build_sequences(test_d, tpf_cached, qid_info)
        d_tpf = len(X_tr[0][0])
        ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
                 max(len(s) for s in X_te))
        print(f'  TabPFN seqs: {len(X_tr)} train / {len(X_va)} val / '
              f'{len(X_te)} test, dim={d_tpf}')

        Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_and_normalize(
            X_tr, y_tr, ml)
        Xn_va, l_va, yn_va, _, _, _, _ = pad_and_normalize(
            X_va, y_va, ml, Xm, Xs, ym, ys)
        Xn_te, l_te, yn_te, _, _, _, _ = pad_and_normalize(
            X_te, y_te, ml, Xm, Xs, ym, ys)

        print(f'\n[Stage 2] Training Bi-LSTM with TabPFN resources...')
        qe_tpf, np_tpf = train_bilstm(
            RatioDataset(Xn_tr, l_tr, yn_tr),
            RatioDataset(Xn_va, l_va, yn_va),
            RatioDataset(Xn_te, l_te, yn_te),
            ym, ys, d_tpf, seed=args.seed, epochs=args.epochs)

        n = len(qe_tpf)
        all_results['TabPFN→BiLSTM'] = {
            'P50': qe_tpf[n//2], 'P90': qe_tpf[int(n*0.9)],
            'P95': qe_tpf[int(n*0.95)], 'params': np_tpf, 'dim': d_tpf
        }

    # ═══ XGBoost Pipeline ═══
    if not args.skip_xgboost:
        print('\n' + '=' * 60)
        print('PIPELINE B: XGBoost → Bi-LSTM')
        print('=' * 60)

        print('\n[Stage 1] Training XGBoost resource predictors...')
        xgb_predict = train_xgboost_predictors(ndv_cache, seed=args.seed)

        xgb_cache = {}
        for qid in unique_qids:
            pred = xgb_predict(qid)
            if pred: xgb_cache[qid] = pred
        print(f'  XGBoost predictions cached: {len(xgb_cache)} queries')

        def xgb_cached(qid): return xgb_cache.get(qid)

        X_tr, y_tr = build_sequences(train_d, xgb_cached, qid_info)
        X_va, y_va = build_sequences(val_d, xgb_cached, qid_info)
        X_te, y_te = build_sequences(test_d, xgb_cached, qid_info)
        d_xgb = len(X_tr[0][0])
        ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
                 max(len(s) for s in X_te))
        print(f'  XGBoost seqs: {len(X_tr)} train / {len(X_va)} val / '
              f'{len(X_te)} test, dim={d_xgb}')

        Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_and_normalize(
            X_tr, y_tr, ml)
        Xn_va, l_va, yn_va, _, _, _, _ = pad_and_normalize(
            X_va, y_va, ml, Xm, Xs, ym, ys)
        Xn_te, l_te, yn_te, _, _, _, _ = pad_and_normalize(
            X_te, y_te, ml, Xm, Xs, ym, ys)

        print(f'\n[Stage 2] Training Bi-LSTM with XGBoost resources...')
        qe_xgb, np_xgb = train_bilstm(
            RatioDataset(Xn_tr, l_tr, yn_tr),
            RatioDataset(Xn_va, l_va, yn_va),
            RatioDataset(Xn_te, l_te, yn_te),
            ym, ys, d_xgb, seed=args.seed, epochs=args.epochs)

        n = len(qe_xgb)
        all_results['XGBoost→BiLSTM'] = {
            'P50': qe_xgb[n//2], 'P90': qe_xgb[int(n*0.9)],
            'P95': qe_xgb[int(n*0.95)], 'params': np_xgb, 'dim': d_xgb
        }

    # ═══ Comparison ═══
    print('\n' + '=' * 60)
    print('FINAL COMPARISON: Stage 1 Resource Predictor → Bi-LSTM')
    print(f'80/10/10 query-level split, seed={args.seed}')
    print('=' * 60)

    print(f'\n  {"Method":<25} {"P50":>7} {"P90":>7} {"P95":>7} '
          f'{"Dim":>5} {"Params":>8}')
    print('  ' + '-' * 58)
    for name, r in all_results.items():
        print(f'  {name:<25} {r["P50"]:.2f}x {r["P90"]:.2f}x '
              f'{r["P95"]:.2f}x {r["dim"]:>5} {r["params"]:>8,}')

    if 'TabPFN→BiLSTM' in all_results and 'XGBoost→BiLSTM' in all_results:
        t = all_results['TabPFN→BiLSTM']
        x = all_results['XGBoost→BiLSTM']
        dp50 = (t['P50'] - x['P50']) / x['P50'] * 100
        dp90 = (t['P90'] - x['P90']) / x['P90'] * 100
        winner = 'TabPFN' if dp50 < 0 else 'XGBoost'
        print(f'\n  TabPFN vs XGBoost: ΔP50={dp50:+.1f}%  '
              f'ΔP90={dp90:+.1f}%  ← {winner} wins!')

    print('=' * 60)


if __name__ == '__main__':
    main()
