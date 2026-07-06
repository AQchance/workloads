"""
ICONQ → Bi-LSTM concurrent runtime prediction (STRICT paper reproduction).

Feature encoding (ICONQ Section 3.1, Figure 3-4):
  Per-query feature (47 dims):
    - 19 operator types × 2 (count, log(1+sum_estRows)) = 38 dims
    - 8 tables × log(1+max_estRows)                    =  8 dims
    - XGBoost-predicted serial latency (Stage proxy)   =  1 dim

  Interaction feature (97 dims = 47+47+3):
    - Qi's 47-dim query features
    - Qj's 47-dim query features (target query)
    - |ti - tj|, 1(ti<tj), 1(tj<ti)

  Prediction target: log(1 + absolute_concurrent_runtime_seconds)
    (NOT slowdown ratio — ICONQ predicts raw system runtime)

Model: Bi-LSTM (Xavier init, BatchNorm, 2 layers) — matches ICONQ repo
Hard split: 80/10/10 query-level, seed=42
"""
import os, re, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = '/home/anqian/Desktop/my_lab/workloads'
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
SEED = 42
EPOCHS = 250

# ═══════════ ICONQ Feature Extraction (Section 3.1) ═══════════

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


def extract_iconq_query_features(qid, xgb_latency_cache):
    """Extract 47-dim ICONQ query feature (Section 3.1)."""
    pf = os.path.join(ROOT, 'explain_plans', f'{qid}.txt')
    if not os.path.exists(pf):
        return None
    with open(pf) as f:
        plan = f.read()

    op_counts = {op: 0 for op in OP_TYPES}
    op_est_rows = {op: 0.0 for op in OP_TYPES}
    table_card = {t: 0.0 for t in TABLES}

    for line in plan.split('\n'):
        if '\t' not in line or line.startswith('--'):
            continue
        parts = line.lstrip(' │├└─').split('\t')
        if len(parts) < 5:
            continue
        raw_op = re.sub(r'^[│├└─\s]+', '', parts[0].strip())
        op = re.sub(r'\(Build\)|\(Probe\)', '', raw_op).strip()
        op = re.sub(r'_\d+$', '', op)
        try:
            est = float(parts[1].strip())
        except ValueError:
            est = 1.0

        if op in op_counts:
            op_counts[op] += 1
            op_est_rows[op] += est
        oi = parts[4].strip() if len(parts) > 4 else ''
        for t in TABLES:
            if t in oi.lower():
                table_card[t] = max(table_card[t], est)

    feat = []
    # 2 × n_p plan features: count + log(1+sum_estRows)
    for op in OP_TYPES:
        feat.append(float(op_counts[op]))
        feat.append(math.log(1 + op_est_rows[op]))
    # n_t table features: log(1+max_estRows)
    for t in TABLES:
        feat.append(math.log(1 + table_card[t]))
    # Runtime feature (Stage proxy → XGBoost latency)
    lat_entry = xgb_latency_cache.get(qid, {})
    lat_log = lat_entry.get('lat', math.log(1 + 10.0))  # fallback 10s
    feat.append(lat_log)

    return np.array(feat, dtype=np.float32)


def build_interaction_vector(qi_feat, qj_feat, ti, tj):
    """97-dim ICONQ interaction vector (Figure 4)."""
    return np.concatenate([
        qi_feat,                           # 47
        qj_feat,                           # 47
        np.array([abs(ti - tj),
                  1.0 if ti < tj else 0.0,
                  1.0 if tj < ti else 0.0], dtype=np.float32),  # 3
    ])


# ═══════════ Build Sequences ═══════════

def build_sequences(data_list, feat_cache, qid_info):
    """
    ICONQ: predict absolute concurrent runtime (seconds).
    Target: log(1 + concurrent_runtime)
    """
    X, y_abs = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if sti == 'penalty' or qi not in feat_cache:
            continue
        qj_feat = feat_cache[qi]  # target query features
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov
                 if oq in qid_info and oq in feat_cache]
        peers.sort()
        for osv, oq in peers:
            qi_feat = feat_cache[oq]  # concurrent query features
            feat = build_interaction_vector(qi_feat, qj_feat, osv, si)
            seq.append(feat)
        if seq:
            X.append(seq)
            y_abs.append(rti)  # absolute concurrent runtime (seconds)
    return X, y_abs


# ═══════════ Bi-LSTM Model (ICONQ repo exact) ═══════════

class ConcurrentDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.lengths[i], self.y[i]


def collate_fn(batch):
    X, l, y = zip(*batch)
    si = torch.argsort(torch.stack(l), descending=True)
    return (torch.stack([X[i] for i in si]),
            torch.stack([l[i] for i in si]),
            torch.stack([y[i] for i in si]))


class IconqBiLSTM(nn.Module):
    """ICONQ repo exact: Xavier init, BatchNorm, no activation in embed/output."""
    def __init__(self, input_dim, embed_dim=128, hidden_size=256, num_layers=2,
                 dropout=0.1):
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
        # Xavier init
        for m in [self.embedding, self.bilstm, self.output_layer]:
            for n, p in m.named_parameters():
                if 'weight' in n:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in n:
                    nn.init.constant_(p.data, 0.0)

    def forward(self, X, lengths):
        # BatchNorm on feature dimension
        if X.shape[1] > 1:
            X = torch.transpose(X, 1, 2)
            X = self.bn(X)
            X = torch.transpose(X, 1, 2)
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Last timestep
        output = output[torch.arange(len(lengths)), lengths - 1]
        return self.output_layer(output).squeeze(-1)


# ═══════════ Training ═══════════

def pad_normalize(X_seq, y_raw, ml, Xm=None, Xs=None, ym=None, ys=None):
    d = len(X_seq[0][0])
    Xa = np.zeros((len(X_seq), ml, d), dtype=np.float32)
    for i, s in enumerate(X_seq):
        Xa[i, :len(s)] = s
    lens = np.array([len(s) for s in X_seq], dtype=np.int32)
    if Xm is None:
        mask = np.zeros_like(Xa)
        for i, s in enumerate(X_seq):
            mask[i, :len(s)] = 1.0
        Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        diff = ((Xa - Xm) * mask) ** 2
        Xs = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
    if ym is None:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
        ym, ys = float(yl.mean()), float(yl.std()) + 1e-8
    else:
        yl = np.log(1 + np.array(y_raw, dtype=np.float32))
    return (Xa - Xm) / Xs, lens, (yl - ym) / ys, Xm, Xs, ym, ys


def train(train_ds, val_ds, test_ds, ym, ys, input_dim):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = IconqBiLSTM(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, l, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(X, l), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    ap.append(model(X.to(device), l).cpu().numpy())
                    at.append(y.numpy())
            pz, tz = np.concatenate(ap), np.concatenate(at)
            # ICONQ evaluates in original seconds space
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0 or epoch == 1:
                print(f'  E{epoch:3d} val={med:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state)
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, l, y in test_loader:
            ap.append(model(X.to(device), l).cpu().numpy())
            at.append(y.numpy())
    pz, tz = np.concatenate(ap), np.concatenate(at)
    pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
    tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    return qe, best_state


# ═══════════ Main ═══════════

def main():
    print('=' * 55)
    print('ICONQ → Bi-LSTM (97-dim, Hard Split)')
    print('=' * 55)

    # Load XGBoost latency cache (as Stage proxy)
    xgb_path = os.path.join(CKPT_DIR, 'oof_xgboost_k5.json')
    with open(xgb_path) as f:
        xgb_cache = json.load(f)
    print(f'[1] XGBoost latency cache: {len(xgb_cache)} queries')

    # Load timeline
    traces = ['collect_concurrent/trace_2_mixed.csv',
              'collect_concurrent/trace_3_fixed_mixed.csv',
              'collect_concurrent/trace_4_fixed_mixed.csv']
    timeline = []
    for tf in traces:
        with open(os.path.join(ROOT, tf)) as f:
            for row in csv.DictReader(f):
                rt = float(row['runtime'])
                actual = 60.0 if row['status'] == 'penalty' else rt
                timeline.append((float(row['start']),
                                 float(row['start']) + actual,
                                 row['qid'], actual, row['status']))
    qid_info = {q: (s, e) for s, e, q, _, _ in timeline}

    with open(os.path.join(ROOT, 'lstm/timeline_qids.txt')) as f:
        valid_qids = set(line.strip() for line in f)
    unique_qids = sorted(set(q for _, _, q, _, _ in timeline if q in valid_qids))
    print(f'[2] Timeline: {len(timeline)} events, {len(unique_qids)} unique queries')

    # Extract ICONQ features
    print(f'[3] Extracting ICONQ features...')
    feat_cache = {}
    for qid in unique_qids:
        f = extract_iconq_query_features(qid, xgb_cache)
        if f is not None:
            feat_cache[qid] = f
    print(f'    Cached: {len(feat_cache)} queries, dim={len(list(feat_cache.values())[0])}')

    # 80/10/10 hard split
    np.random.seed(SEED)
    np.random.shuffle(unique_qids)
    n_tr = int(len(unique_qids) * 0.8)
    n_va = int(len(unique_qids) * 0.1)
    train_qids = set(unique_qids[:n_tr])
    val_qids = set(unique_qids[n_tr:n_tr + n_va])
    test_qids = set(unique_qids[n_tr + n_va:])
    print(f'[4] Split: {len(train_qids)}/{len(val_qids)}/{len(test_qids)}')

    # Build event list
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(timeline):
        ov = [qj for j, (sj, ej, qj, _, _) in enumerate(timeline)
              if i != j and sj < ei and ej > si]
        ci.append((qi, si, ei, rti, sti, ov))
    train_d = [c for c in ci if c[0] in train_qids]
    val_d = [c for c in ci if c[0] in val_qids]
    test_d = [c for c in ci if c[0] in test_qids]

    X_tr, y_tr = build_sequences(train_d, feat_cache, qid_info)
    X_va, y_va = build_sequences(val_d, feat_cache, qid_info)
    X_te, y_te = build_sequences(test_d, feat_cache, qid_info)
    d_in = len(X_tr[0][0])
    ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_va),
             max(len(s) for s in X_te))
    print(f'[5] Seqs: {len(X_tr)}/{len(X_va)}/{len(X_te)}, max_len={ml}, dim={d_in}')

    Xn_tr, l_tr, yn_tr, Xm, Xs, ym, ys = pad_normalize(X_tr, y_tr, ml)
    Xn_va, l_va, yn_va, _, _, _, _ = pad_normalize(X_va, y_va, ml, Xm, Xs, ym, ys)
    Xn_te, l_te, yn_te, _, _, _, _ = pad_normalize(X_te, y_te, ml, Xm, Xs, ym, ys)

    print(f'[6] Training...')
    qe, best_state = train(ConcurrentDataset(Xn_tr, l_tr, yn_tr),
               ConcurrentDataset(Xn_va, l_va, yn_va),
               ConcurrentDataset(Xn_te, l_te, yn_te), ym, ys, d_in)

    # Save model + norm stats for scheduler
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    model_path = os.path.join(ckpt_dir, 'iconq_bilstm.pt')
    torch.save(best_state, model_path)
    norm_path = os.path.join(ckpt_dir, 'iconq_norm.npz')
    np.savez(norm_path, X_mean=Xm, X_std=Xs, y_mean=ym, y_std=ys)
    print(f'  Saved: {model_path}')
    print(f'  Saved: {norm_path}')

    n = len(qe)
    print(f'\n{"="*55}')
    print(f'ICONQ → BiLSTM (absolute runtime):')
    for pct, label in [(10, 'P10'), (50, 'P50'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
        print(f'  {label}: {qe[min(int(n*pct/100), n-1)]:.2f}x')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
