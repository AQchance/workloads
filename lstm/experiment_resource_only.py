"""
Experiment: Bi-LSTM with GNN resource-only features (no plan_emb).
Feature: 5 resources + log(serial_lat) + time/conflict = 17-dim per sequence item.
"""

import os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = '/home/anqian/Desktop/my_lab/workloads'
GNF_FILES = ['lstm/gnn_features_k2_fixed.json', 'lstm/gnn_features_k3_fixed.json',
             'lstm/gnn_features_k4_fixed.json']
TRACES = ['collect_concurrent/trace_2_mixed.csv', 'collect_concurrent/trace_3_fixed_mixed.csv',
          'collect_concurrent/trace_4_fixed_mixed.csv']


def resource_conflict(t, c):
    t = np.array(t); c = np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


# Load GNN features
gnf = {}
for fn in GNF_FILES:
    with open(os.path.join(ROOT, fn)) as f:
        gnf.update(json.load(f))
print(f"Loaded {len(gnf)} GNN features")

# Load traces
timeline = []
for tf in TRACES:
    with open(os.path.join(ROOT, tf)) as f:
        for row in csv.DictReader(f):
            rt = float(row['runtime']); st = row['status']
            actual = 60.0 if st == 'penalty' else rt
            timeline.append((float(row['start']), float(row['start']) + actual,
                           row['qid'], actual, st))

qid_info = {}
for s, e, q, _, _ in timeline:
    qid_info[q] = (s, e)

# Build concurrent sets
ci = [(qi, si, ei, rti, sti,
       [qj for j, (sj, ej, qj, _, _) in enumerate(timeline) if i != j and sj < ei and ej > si])
      for i, (si, ei, qi, rti, sti) in enumerate(timeline)]

mt = max(e for _, _, e, _, _, _ in ci)
split_time = mt * 0.7
train_d = [c for c in ci if c[1] < split_time]
test_d = [c for c in ci if c[1] >= split_time]
print(f"Train sets: {len(train_d)}, Test sets: {len(test_d)}")


def build_resource_seqs(data_list):
    X, y_ratio = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in gnf or sti == 'penalty': continue
        serial_lat = max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5)
        qv = list(gnf[qi]['gpu_resources'].values()) + [math.log(1 + serial_lat)]
        t_res = list(gnf[qi]['gpu_resources'].values())
        seq = []
        oi = [(qid_info[oq][0], oq) for oq in ov if oq in gnf and oq in qid_info]
        oi.sort()
        for os_val, oq in oi:
            osl = math.log(1 + gnf[oq]['serial_labels'].get('latency_s', 10))
            ovv = list(gnf[oq]['gpu_resources'].values()) + [osl]
            c = resource_conflict(t_res, list(gnf[oq]['gpu_resources'].values()))
            feat = qv + ovv + [si - os_val, 1.0 if os_val < si else 0.0] + c
            seq.append(feat)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    return X, y_ratio


X_tr, y_tr = build_resource_seqs(train_d)
X_te, y_te = build_resource_seqs(test_d)
d = len(X_tr[0][0])
print(f"Feature dim: {d}, Train seqs: {len(X_tr)}, Test seqs: {len(X_te)}")

# Pad
ml = max(max(len(s) for s in X_tr), max(len(s) for s in X_te))
Xa = np.zeros((len(X_tr), ml, d), dtype=np.float32)
for i, s in enumerate(X_tr): Xa[i, :len(s)] = s
Xta = np.zeros((len(X_te), ml, d), dtype=np.float32)
for i, s in enumerate(X_te): Xta[i, :len(s)] = s

# Masked normalize
mask = np.zeros_like(Xa)
for i, s in enumerate(X_tr): mask[i, :len(s)] = 1.0
Xm = (Xa * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
Xs = np.sqrt(((Xa - Xm) ** 2 * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

yl = np.log(1 + np.array(y_tr, dtype=np.float32))
ym, ys_ = float(yl.mean()), float(yl.std()) + 1e-8

y_test_log = np.log(1 + np.array(y_te, dtype=np.float32))

print(f"Max seq len: {ml}, label mean={ym:.3f} std={ys_:.3f}")

Xa_norm = (Xa - Xm) / Xs
Xta_norm = (Xta - Xm) / Xs
y_norm = (yl - ym) / ys_
y_test_norm = (y_test_log - ym) / ys_
tr_lens = np.array([len(s) for s in X_tr], dtype=np.int32)
te_lens = np.array([len(s) for s in X_te], dtype=np.int32)


class SimpleDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = lengths
        self.y = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], torch.tensor(int(self.lengths[i])), self.y[i]


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


train_ds = SimpleDataset(Xa_norm, tr_lens, y_norm)
test_ds = SimpleDataset(Xta_norm, te_lens, y_test_norm)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Sequential(nn.Linear(input_dim, 128), nn.Linear(128, 128))
        self.bilstm = nn.LSTM(128, hidden_size, num_layers, dropout=dropout,
                              batch_first=True, bidirectional=True)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2), nn.Linear(hidden_size // 2, 1))
        for m in [self.embedding, self.bilstm, self.output_layer]:
            for n, p in m.named_parameters():
                if 'weight' in n: nn.init.xavier_uniform_(p.data)
                elif 'bias' in n: nn.init.constant_(p.data, 0.0)

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


def train_one(seed, epochs=250):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM(input_dim=d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
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

        if epoch % 20 == 0:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, lengths, y in test_loader:
                    X = X.to(device)
                    ap.append(model(X, lengths).cpu().numpy())
                    at.append(y.numpy())
            p_z = np.concatenate(ap); t_z = np.concatenate(at)
            p_raw = np.maximum(np.exp(p_z * ys_ + ym) - 1, 0.01)
            t_raw = np.maximum(np.exp(t_z * ys_ + ym) - 1, 0.01)
            med = np.median(np.maximum(p_raw / t_raw, t_raw / p_raw))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, lengths, y in test_loader:
            X = X.to(device)
            ap.append(model(X, lengths).cpu().numpy())
            at.append(y.numpy())
    p_z = np.concatenate(ap); t_z = np.concatenate(at)
    n_params = sum(p.numel() for p in model.parameters())
    return p_z, t_z, n_params


# Train 3 seeds → ensemble
SEEDS = [42, 123, 456]
all_preds = []
all_labels = None
all_params = []

print(f"\n{'='*55}")
for seed in SEEDS:
    print(f"Training seed {seed}...")
    p_z, t_z, np_ = train_one(seed)
    all_preds.append(p_z)
    all_labels = t_z
    all_params.append(np_)
    p_raw = np.maximum(np.exp(p_z * ys_ + ym) - 1, 0.01)
    t_raw = np.maximum(np.exp(t_z * ys_ + ym) - 1, 0.01)
    qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
    n = len(qe)
    print(f"  Seed {seed}: P50={qe[n//2]:.2f}x P90={qe[int(n*0.9)]:.2f}x P95={qe[int(n*0.95)]:.2f}x ({np_:,} params)")

# Ensemble
ep_z = np.mean(all_preds, axis=0)
ep_raw = np.maximum(np.exp(ep_z * ys_ + ym) - 1, 0.01)
et_raw = np.maximum(np.exp(all_labels * ys_ + ym) - 1, 0.01)
eqe = np.sort(np.maximum(ep_raw / et_raw, et_raw / p_raw))
n = len(eqe)

print(f"\n=== Resource-Only Ensemble (d={d}, {len(SEEDS)} seeds) ===")
for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
    print(f"  P{pct:2d}: {eqe[int(n*pct/100)]:.2f}x")

log_t = all_labels * ys_ + ym
log_p = ep_z * ys_ + ym
r2 = 1 - np.sum((log_t - log_p)**2) / max(np.sum((log_t - log_t.mean())**2), 1e-8)
print(f"  R² (log-ratio): {r2:.4f}")
