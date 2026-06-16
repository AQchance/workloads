"""
Query-Level Split Experiment: ICONQ baseline vs GNN ResourceFull BiLSTM.
Train/test split by query ID (70/30) — same query never appears in both sets.
Reproducible results: seed=42, 250 epochs.

Usage:
    cd /home/anqian/Desktop/my_lab/workloads
    source /home/anqian/code/python/workloads/venv/bin/activate
    python lstm/query_split_exp.py
    python lstm/query_split_exp.py --seed 42 --epochs 250
"""

import os, sys, json, csv, math, re, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

ROOT = '/home/anqian/Desktop/my_lab/workloads'
OUT_DIR = os.path.join(ROOT, 'lstm', 'query_split_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Constants ───
OP_TYPES = ['TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
            'IndexLookUp', 'IndexReader', 'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
            'HashAgg', 'StreamAgg', 'Sort', 'TopN', 'Window',
            'ExchangeSender', 'ExchangeReceiver', 'Projection', 'Selection']
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer', 'nation', 'region']

GNF_FILES = ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
             'lstm/gnn_features_k4_fixed.json']
TRACE_FILES = ['collect_concurrent/trace_2_mixed.csv', 'collect_concurrent/trace_3_fixed_mixed.csv',
               'collect_concurrent/trace_4_fixed_mixed.csv']


# ═══════════════════════ Data Loading ═══════════════════════

def load_gnn_features():
    gnf = {}
    for fn in GNF_FILES:
        with open(os.path.join(ROOT, fn)) as f:
            gnf.update(json.load(f))
    return gnf


def load_timeline():
    timeline = []
    for tf in TRACE_FILES:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime'])
                actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']), float(row['start']) + actual,
                                 row['qid'], actual, row['status']))
    return timeline


def build_concurrent_sets(timeline):
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    return ci


# ═══════════════════════ ICONQ Feature Extraction ═══════════════════════

def build_iconq_features(gnf):
    iconq = {}
    for qid in gnf:
        pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
        if not os.path.exists(pf):
            continue
        with open(pf) as f:
            plan = f.read()
        pred_lat = max(abs(float(gnf[qid]['gpu_resources'].get('lat', 1))), 0.5)
        op_counts = {o: 0 for o in OP_TYPES}
        op_estrows = {o: 0.0 for o in OP_TYPES}
        table_estrows = {t: 0.0 for t in TABLES}
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
            if op in op_counts:
                op_counts[op] += 1
                op_estrows[op] += est
            table_info = parts[4].strip() if len(parts) > 4 else ''
            for t in TABLES:
                if t in table_info.lower():
                    table_estrows[t] = max(table_estrows[t], est)
        feat = [math.log(1 + pred_lat)]
        for o in OP_TYPES:
            feat.append(float(op_counts[o]))
            feat.append(math.log(1 + op_estrows[o]))
        for t in TABLES:
            feat.append(math.log(1 + table_estrows[t]))
        iconq[qid] = feat
    return iconq


# ═══════════════════════ Sequence Builders ═══════════════════════

def _resource_conflict(t, c):
    t_arr, c_arr = np.array(t), np.array(c)
    return list(np.minimum(t_arr, c_arr) / np.maximum(np.abs(t_arr) + np.abs(c_arr) + 1e-8, 1e-8))


def build_iconq_sequences(data_list, iconq_feats, qid_info):
    X, y_abs = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in iconq_feats or sti == 'penalty':
            continue
        qv = iconq_feats[qi]
        peers = [(qid_info[oq][0], oq) for oq in ov if oq in iconq_feats and oq in qid_info]
        if not peers:
            continue
        peers.sort()
        seq = [qv + iconq_feats[oq] + [si - osv, 1.0 if osv < si else 0.0] for osv, oq in peers]
        if seq:
            X.append(seq)
            y_abs.append(rti)
    return X, y_abs


def build_gnn_sequences(data_list, gnf, qid_info):
    X, y_ratio = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty':
            continue
        sl_ = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = gnf[qi]['plan_emb'] + list(gnf[qi]['gpu_resources'].values()) + [math.log(1 + sl_)]
        tr_ = list(gnf[qi]['gpu_resources'].values())
        peers = [(qid_info[oq][0], oq) for oq in ov if oq in gnf and oq in qid_info]
        peers.sort()
        seq = []
        for osv, oq in peers:
            oslv = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = gnf[oq]['plan_emb'] + list(gnf[oq]['gpu_resources'].values()) + [oslv]
            c = _resource_conflict(tr_, list(gnf[oq]['gpu_resources'].values()))
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / sl_)
    return X, y_ratio


# ═══════════════════════ Data Normalization ═══════════════════════

class RatioDataset(torch.utils.data.Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.lengths[i], self.y[i]


class AbsDataset(torch.utils.data.Dataset):
    def __init__(self, X, lengths, y, y_mean, y_std):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.lengths[i], self.y[i]


def make_dense_and_normalize(X_seq, y_raw, ml, label_space='abs'):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq):
        Xa[i, :len(s)] = s
    mask = np.zeros_like(Xa)
    for i, s in enumerate(X_seq):
        mask[i, :len(s)] = 1.0
    Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
    Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    X_norm = (Xa - Xm) / Xs
    y_norm = (yl - ym) / ys
    if label_space == 'abs':
        return AbsDataset(X_norm, lens, y_norm, ym, ys)
    else:
        return RatioDataset(X_norm, lens, y_norm), ym, ys


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


# ═══════════════════════ Models ═══════════════════════

class ICONQBiLSTM(nn.Module):
    """ICONQ paper exact reproduction: BiLSTM with Xavier init, BatchNorm, Adam."""
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Linear(128, 128))
        self.bilstm = nn.LSTM(128, hidden_size, num_layers, dropout=dropout,
                              batch_first=True, bidirectional=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2), nn.Linear(hidden_size // 2, 1))
        for module in [self.embedding, self.bilstm, self.output]:
            for n, p in module.named_parameters():
                if 'weight' in n:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in n:
                    nn.init.constant_(p.data, 0.0)

    def forward(self, x, lengths):
        if x.shape[1] > 1:
            x = torch.transpose(x, 1, 2)
            x = self.bn(x)
            x = torch.transpose(x, 1, 2)
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.output(output[torch.arange(len(lengths)), lengths - 1]).squeeze(-1)


class ResourceFullBiLSTM(nn.Module):
    """GNN plan_emb + Resource Gate (per-timestep) + Resource Bias (output)."""
    def __init__(self, input_dim=275, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.res_gate = nn.Sequential(
            nn.Linear(10, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim), nn.Sigmoid())
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))
        self.res_bias = nn.Sequential(
            nn.Linear(10, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1))

    def forward(self, X, lengths):
        res_pairs = torch.cat([X[:, :, 128:133], X[:, :, 262:267]], dim=-1)
        X_gated = X * self.res_gate(res_pairs)
        x = self.embedding(X_gated)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.bilstm(packed)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)
        base_pred = self.predictor(final).squeeze(-1)
        bias = self.res_bias(res_pairs.mean(dim=1)).squeeze(-1)
        return base_pred + bias


# ═══════════════════════ Training ═══════════════════════

def train_iconq(train_ds, test_ds, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = ICONQBiLSTM(input_dim=train_ds.X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None
    y_std, y_mean = test_ds.y_std, test_ds.y_mean

    for epoch in range(1, epochs + 1):
        model.train()
        for X, lengths, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, lengths), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, lengths, y in test_loader:
                    X = X.to(device)
                    ap.append(model(X, lengths).cpu().numpy())
                    at.append(y.numpy())
            p_z = np.concatenate(ap)
            t_z = np.concatenate(at)
            p_sec = np.maximum(np.exp(p_z * y_std + y_mean) - 1, 0.01)
            t_sec = np.maximum(np.exp(t_z * y_std + y_mean) - 1, 0.01)
            med = np.median(np.maximum(p_sec / t_sec, t_sec / p_sec))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  ICONQ E{epoch:3d} best={best_val:.2f}x cur={med:.2f}x')

    model.load_state_dict(best_state)
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, lengths, y in test_loader:
            X = X.to(device)
            ap.append(model(X, lengths).cpu().numpy())
            at.append(y.numpy())
    p_z = np.concatenate(ap)
    t_z = np.concatenate(at)
    p_sec = np.maximum(np.exp(p_z * y_std + y_mean) - 1, 0.01)
    t_sec = np.maximum(np.exp(t_z * y_std + y_mean) - 1, 0.01)
    qe = np.maximum(p_sec / t_sec, t_sec / p_sec)
    n_params = sum(p.numel() for p in model.parameters())
    return model, best_state, qe, p_sec, t_sec, n_params


def train_gnn(train_ds, test_ds, ym, ys, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = ResourceFullBiLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        for X, lengths, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, lengths), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, lengths, y in test_loader:
                    X = X.to(device)
                    ap.append(model(X, lengths).cpu().numpy())
                    at.append(y.numpy())
            p_z = np.concatenate(ap)
            t_z = np.concatenate(at)
            p_raw = np.maximum(np.exp(p_z * ys + ym) - 1, 0.01)
            t_raw = np.maximum(np.exp(t_z * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(p_raw / t_raw, t_raw / p_raw))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  GNN   E{epoch:3d} best={best_val:.2f}x cur={med:.2f}x')

    model.load_state_dict(best_state)
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, lengths, y in test_loader:
            X = X.to(device)
            ap.append(model(X, lengths).cpu().numpy())
            at.append(y.numpy())
    p_z = np.concatenate(ap)
    t_z = np.concatenate(at)
    p_raw = np.maximum(np.exp(p_z * ys + ym) - 1, 0.01)
    t_raw = np.maximum(np.exp(t_z * ys + ym) - 1, 0.01)
    qe = np.maximum(p_raw / t_raw, t_raw / p_raw)
    n_params = sum(p.numel() for p in model.parameters())
    return model, best_state, qe, p_raw, t_raw, n_params


# ═══════════════════════ Evaluation ═══════════════════════

def compute_metrics(qe, n):
    qe_sorted = np.sort(qe)
    metrics = {}
    for pct in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        idx = min(int(n * pct / 100), n - 1)
        metrics[f'P{pct}'] = round(float(qe_sorted[idx]), 2)
    return metrics


def print_results(name, qe, n, n_params, label_type):
    qe_sorted = np.sort(qe)
    print(f'\n{name} ({n_params:,} params, {label_type})')
    print(f'  P10={qe_sorted[int(n*0.10)]:.2f}x  P25={qe_sorted[int(n*0.25)]:.2f}x  '
          f'P50={qe_sorted[n//2]:.2f}x  P75={qe_sorted[int(n*0.75)]:.2f}x  '
          f'P90={qe_sorted[int(n*0.90)]:.2f}x  P95={qe_sorted[int(n*0.95)]:.2f}x')


# ═══════════════════════ Main ═══════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--split-ratio', type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Seed: {args.seed}, Epochs: {args.epochs}, Split: {args.split_ratio}')

    # ─── Load data ───
    print(f'\n{"="*60}\nLoading data...')
    gnf = load_gnn_features()
    print(f'  GNN features: {len(gnf)} queries')

    timeline = load_timeline()
    print(f'  Timeline: {len(timeline)} events')

    # ─── Query-level split ───
    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in gnf))
    np.random.seed(args.seed)
    np.random.shuffle(unique_qids)
    n_train_q = int(len(unique_qids) * args.split_ratio)
    train_qids = set(unique_qids[:n_train_q])
    test_qids = set(unique_qids[n_train_q:])
    print(f'  Query split: {len(train_qids)} train / {len(test_qids)} test')

    ci = build_concurrent_sets(timeline)
    train_d = [c for c in ci if c[0] in train_qids]
    test_d = [c for c in ci if c[0] in test_qids]
    print(f'  Events: {len(train_d)} train / {len(test_d)} test')

    qid_info = {}
    for s, e, q, _, _ in timeline:
        qid_info[q] = (s, e)

    # ─── Build features ───
    iconq_feats = build_iconq_features(gnf)
    print(f'  ICONQ features: {len(iconq_feats)} queries')

    X_tr_ic, y_tr_ic = build_iconq_sequences(train_d, iconq_feats, qid_info)
    X_te_ic, y_te_ic = build_iconq_sequences(test_d, iconq_feats, qid_info)

    X_tr_gnn, y_tr_gnn = build_gnn_sequences(train_d, gnf, qid_info)
    X_te_gnn, y_te_gnn = build_gnn_sequences(test_d, gnf, qid_info)

    ml_ic = max(max(len(s) for s in X_tr_ic), max(len(s) for s in X_te_ic))
    ml_gnn = max(max(len(s) for s in X_tr_gnn), max(len(s) for s in X_te_gnn))

    print(f'\n  ICONQ: {len(X_tr_ic)} train / {len(X_te_ic)} test, '
          f'dim={len(X_tr_ic[0][0])}, max_len={ml_ic}')
    print(f'  GNN:   {len(X_tr_gnn)} train / {len(X_te_gnn)} test, '
          f'dim={len(X_tr_gnn[0][0])}, max_len={ml_gnn}')

    # ─── Normalize ───
    train_ds_ic = make_dense_and_normalize(X_tr_ic, y_tr_ic, ml_ic, 'abs')
    test_ds_ic = make_dense_and_normalize(X_te_ic, y_te_ic, ml_ic, 'abs')

    train_ds_gnn, ym_g, ys_g = make_dense_and_normalize(X_tr_gnn, y_tr_gnn, ml_gnn, 'ratio')
    test_ds_gnn, _, _ = make_dense_and_normalize(X_te_gnn, y_te_gnn, ml_gnn, 'ratio')

    # ─── Train ICONQ Baseline ───
    print(f'\n{"="*60}\nTraining ICONQ Baseline (seed={args.seed})...')
    model_ic, state_ic, qe_ic, p_ic, t_ic, np_ic = train_iconq(
        train_ds_ic, test_ds_ic, args.seed, args.epochs, device)
    n_ic = len(qe_ic)
    qe_ic_sorted = np.sort(qe_ic)
    print_results('ICONQ Baseline', qe_ic, n_ic, np_ic, 'absolute runtime')

    # ─── Train GNN ResourceFull ───
    print(f'\n{"="*60}\nTraining GNN ResourceFull (seed={args.seed})...')
    model_gnn, state_gnn, qe_gnn, p_gnn, t_gnn, np_gnn = train_gnn(
        train_ds_gnn, test_ds_gnn, ym_g, ys_g, args.seed, args.epochs, device)
    n_gnn = len(qe_gnn)
    qe_gnn_sorted = np.sort(qe_gnn)
    print_results('GNN ResourceFull', qe_gnn, n_gnn, np_gnn, 'slowdown ratio')

    # ─── Results ───
    print(f'\n{"="*60}')
    print('QUERY-LEVEL SPLIT RESULTS (70/30 by query ID)')
    print(f'Seed={args.seed}, Epochs={args.epochs}')
    print(f'{"="*60}')
    print(f'{"Method":<24} {"P50":>6} {"P90":>6} {"P95":>6} {"Params":>10} {"Test":>8}')
    print('-' * 58)
    print(f'{"ICONQ Baseline":<24} {qe_ic_sorted[n_ic//2]:.2f}x '
          f'{qe_ic_sorted[int(n_ic*0.9)]:.2f}x '
          f'{qe_ic_sorted[int(n_ic*0.95)]:.2f}x {np_ic:>10,} {n_ic:>8}')
    print(f'{"GNN ResourceFull":<24} {qe_gnn_sorted[n_gnn//2]:.2f}x '
          f'{qe_gnn_sorted[int(n_gnn*0.9)]:.2f}x '
          f'{qe_gnn_sorted[int(n_gnn*0.95)]:.2f}x {np_gnn:>10,} {n_gnn:>8}')

    # Full distribution
    print(f'\n{"="*60}')
    print('FULL Q-ERROR DISTRIBUTION')
    print(f'{"="*60}')
    print(f'{"Percentile":<12} {"ICONQ":>10} {"GNN_ResourceFull":>18} {"Delta":>10}')
    print('-' * 52)
    for pct in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        i_ic = min(int(n_ic * pct / 100), n_ic - 1)
        i_gnn = min(int(n_gnn * pct / 100), n_gnn - 1)
        q_ic = qe_ic_sorted[i_ic]
        q_gnn = qe_gnn_sorted[i_gnn]
        delta = (q_ic - q_gnn) / q_ic * 100
        print(f'  P{pct:<9} {q_ic:>8.2f}x {q_gnn:>14.2f}x {delta:>+9.1f}%')

    # R²
    log_t_ic = np.log(1 + t_ic)
    log_p_ic = np.log(1 + p_ic)
    r2_ic = 1 - np.sum((log_t_ic - log_p_ic)**2) / max(np.sum((log_t_ic - log_t_ic.mean())**2), 1e-8)

    log_t_gnn = np.log(1 + t_gnn)
    log_p_gnn = np.log(1 + p_gnn)
    r2_gnn = 1 - np.sum((log_t_gnn - log_p_gnn)**2) / max(np.sum((log_t_gnn - log_t_gnn.mean())**2), 1e-8)

    print(f'\n  R² (log-runtime): ICONQ={r2_ic:.4f}  GNN={r2_gnn:.4f}')

    # ─── Save everything ───
    results = {
        'config': {'seed': args.seed, 'epochs': args.epochs, 'split_ratio': args.split_ratio,
                   'split_type': 'query_id', 'device': str(device)},
        'data': {
            'num_queries': len(unique_qids),
            'train_queries': len(train_qids), 'test_queries': len(test_qids),
            'train_events': len(train_d), 'test_events': len(test_d),
            'iconq_train_seqs': len(X_tr_ic), 'iconq_test_seqs': len(X_te_ic),
            'iconq_dim': len(X_tr_ic[0][0]),
            'gnn_train_seqs': len(X_tr_gnn), 'gnn_test_seqs': len(X_te_gnn),
            'gnn_dim': len(X_tr_gnn[0][0]),
        },
        'iconq': {
            'params': int(np_ic),
            'metrics': compute_metrics(qe_ic, n_ic),
            'r2_log_runtime': float(round(r2_ic, 4)),
        },
        'gnn': {
            'params': int(np_gnn),
            'metrics': compute_metrics(qe_gnn, n_gnn),
            'r2_log_runtime': float(round(r2_gnn, 4)),
        },
    }

    results_path = os.path.join(OUT_DIR, f'results_s{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {results_path}')

    # Save models
    torch.save(state_ic, os.path.join(OUT_DIR, f'iconq_baseline_s{args.seed}.pt'))
    torch.save(state_gnn, os.path.join(OUT_DIR, f'gnn_resourcefull_s{args.seed}.pt'))
    print(f'Models saved: {OUT_DIR}/{{iconq_baseline,gnn_resourcefull}}_s{args.seed}.pt')


if __name__ == '__main__':
    main()
