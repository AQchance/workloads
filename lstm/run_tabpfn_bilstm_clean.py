"""
Clean hard-split Bi-LSTM: TabPFN vs XGBoost (NO data leak).

Stage 1: Train TabPFN/XGBoost ONLY on cgroup queries in BiLSTM TRAIN split.
Stage 2: Out-of-sample resource predictions for val/test queries → BiLSTM.

Same 80/10/10 query-level split for BOTH stages.
"""
import os, sys, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = '/home/anqian/Desktop/my_lab/workloads'
sys.path.insert(0, os.path.join(ROOT, 'gnn'))
from train_cgroup import load_cgroup_labels, extract_cpu_resource
from train_ndv import load_ndv_cache, parse_plan
from tabpfn import TabPFNRegressor
from sklearn.ensemble import GradientBoostingRegressor

DIM_NAMES = ['mem', 'disk', 'net', 'lat', 'cpures']


def extract_flat(qid, ndv_cache):
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


def build_stage1_predictor(cgroup_labels, train_qids, ndv_cache, seed, use_tabpfn):
    """
    Train Stage 1 resource predictor ONLY on cgroup queries in train_qids.
    Returns: predict(qid) -> {dim: log(1+value)}
    """
    X_tr, y_tr = [], []
    for qid in train_qids:
        if qid not in cgroup_labels: continue
        lab = cgroup_labels[qid]
        feat = extract_flat(qid, ndv_cache)
        if feat is None: continue
        X_tr.append(feat)
        y_tr.append([
            math.log(1 + max(lab['memory_bytes'], 1)),
            math.log(1 + max(lab['disk_bytes'], 1)),
            math.log(1 + max(lab['network_bytes'], 1)),
            math.log(1 + max(lab['latency_s'], 0.001)),
            math.log(1 + max(lab.get('cpu_resource_s', 0.001), 0.001)),
        ])

    X_tr = np.array(X_tr, dtype=np.float64)
    y_tr = np.array(y_tr, dtype=np.float64)
    print(f'  Stage-1 train: {len(X_tr)} cgroup queries in train split, '
          f'{X_tr.shape[1]} features')

    if use_tabpfn:
        models = []
        for i, dim in enumerate(DIM_NAMES):
            m = TabPFNRegressor(n_estimators=8, random_state=seed,
                                fit_mode='fit_preprocessors', device='cuda')
            m.fit(X_tr.astype(np.float32), y_tr[:, i].astype(np.float32))
            models.append(m)

        def predict(qid):
            feat = extract_flat(qid, ndv_cache)
            if feat is None: return None
            f32 = feat.astype(np.float32).reshape(1, -1)
            return {DIM_NAMES[i]: float(models[i].predict(f32)[0])
                    for i in range(len(DIM_NAMES))}
    else:
        Xm = X_tr.mean(axis=0); Xs = X_tr.std(axis=0) + 1e-8
        X_norm = (X_tr - Xm) / Xs
        models = []
        for i in range(len(DIM_NAMES)):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                           learning_rate=0.1, random_state=seed,
                                           subsample=0.8)
            m.fit(X_norm, y_tr[:, i])
            models.append(m)

        def predict(qid):
            feat = extract_flat(qid, ndv_cache)
            if feat is None: return None
            f_n = (feat - Xm) / Xs
            return {DIM_NAMES[i]: float(models[i].predict(f_n.reshape(1, -1))[0])
                    for i in range(len(DIM_NAMES))}

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


def train_bilstm(train_ds, val_ds, test_ds, ym, ys, input_dim, label, seed=42, epochs=250):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = BiLSTM(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  [{label}] E{epoch:3d} val={med:.2f}x best={best_val:.2f}x')

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
    args = parser.parse_args()

    print('=' * 60)
    print('TabPFN vs XGBoost → Bi-LSTM (Clean: NO data leak)')
    print(f'Seed={args.seed}, Epochs={args.epochs}')
    print('=' * 60)

    ndv_cache = load_ndv_cache(os.path.join(ROOT, 'ndv_cache.json'))

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
                                 row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    # Load GNN features (known query set for timeline)
    gnf = {}
    for fn in ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
               'lstm/gnn_features_k4_fixed.json']:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))

    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))
    print(f'Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # ===== 80/10/10 query-level split (used for BOTH stages) =====
    np.random.seed(args.seed)
    np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8)
    n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr])
    val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'Split: {len(train_qids)} train / {len(val_qids)} val / {len(test_qids)} test')

    # Load all cgroup labels for Stage 1 training (CPU resource from ANALYZE)
    cgroup_labels = load_cgroup_labels(os.path.join(ROOT, 'cgroup_resources'))
    analyze_dir = os.path.join(ROOT, 'explain_analyze_results')
    for qid in cgroup_labels:
        af = os.path.join(analyze_dir, f'{qid}.txt')
        if os.path.exists(af):
            with open(af) as f:
                cgroup_labels[qid]['cpu_resource_s'] = extract_cpu_resource(f.read())

    # Build concurrent event list
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]

    all_results = {}

    for model_name in ['TabPFN', 'XGBoost']:
        print('\n' + '=' * 60)
        print(f'PIPELINE: {model_name} → Bi-LSTM (clean split)')
        print('=' * 60)

        # Stage 1: Train resource predictor ONLY on train_qids cgroup data
        print(f'\n[Stage 1] {model_name} resource predictor (train-qids only)...')
        use_tpf = (model_name == 'TabPFN')
        predictor = build_stage1_predictor(cgroup_labels, train_qids, ndv_cache,
                                           seed=args.seed, use_tabpfn=use_tpf)

        # Stage 2: Build sequences (ALL queries get out-of-sample predictions)
        print(f'\n[Stage 2] Building concurrent sequences...')
        X_tr, y_tr = build_sequences(train_d, predictor, qid_info)
        X_va, y_va = build_sequences(val_d, predictor, qid_info)
        X_te, y_te = build_sequences(test_d, predictor, qid_info)
        d_in = len(X_tr[0][0])
        ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
                 max(len(s) for s in X_te))
        print(f'  Seqs: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test, '
              f'max_len={ml}, dim={d_in}')

        Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = \
            pad_and_normalize(X_tr, y_tr, ml)
        Xn_va, l_va, yn_va, _, _, _, _ = \
            pad_and_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
        Xn_te, l_te, yn_te, _, _, _, _ = \
            pad_and_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

        # Stage 3: Bi-LSTM
        print(f'\n[Stage 3] Training Bi-LSTM...')
        qe, n_params = train_bilstm(
            RatioDataset(Xn_tr, l_tr, yn_tr),
            RatioDataset(Xn_va, l_va, yn_va),
            RatioDataset(Xn_te, l_te, yn_te),
            ym, ys, d_in, model_name, seed=args.seed, epochs=args.epochs)

        n = len(qe)
        all_results[model_name] = {
            'P50': qe[n//2], 'P90': qe[int(n*0.9)], 'P95': qe[int(n*0.95)],
            'P99': qe[int(n*0.99)]
        }

    # ═══ Comparison ═══
    print('\n' + '=' * 60)
    print('FINAL COMPARISON: Clean split, NO data leak')
    print(f'80/10/10 split for BOTH Stage 1 and Stage 2')
    print(f'Seed={args.seed}, Epochs={args.epochs}')
    print('=' * 60)

    print(f'\n  {"Method":<22} {"P50":>7} {"P90":>7} {"P95":>7} {"P99":>7}')
    print('  ' + '-' * 49)
    for name in ['TabPFN', 'XGBoost']:
        r = all_results[name]
        print(f'  {name+"->BiLSTM":<22} {r["P50"]:.2f}x {r["P90"]:.2f}x '
              f'{r["P95"]:.2f}x {r["P99"]:.2f}x')

    t = all_results['TabPFN']; x = all_results['XGBoost']
    winner = 'TabPFN' if t['P50'] < x['P50'] else 'XGBoost'
    for k in ['P50', 'P90', 'P95', 'P99']:
        d = (t[k] - x[k]) / x[k] * 100
        print(f'  Δ{k}={d:+.1f}%', end='')
    print(f'\n  Winner: {winner}')
    print('=' * 60)


if __name__ == '__main__':
    main()
