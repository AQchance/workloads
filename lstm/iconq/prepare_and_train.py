"""
Build training data with ICONQ-style features and train Bi-LSTM.
Compares prediction accuracy against our GNN-based approach.
"""

import sys, os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_bilstm import ConcurrentBiLSTM, collate_fn

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/iconq/iconq_features.json'
TRACE_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm/iconq'

# ─── Load ICONQ features ───
with open(FEATURES_FILE) as f:
    iconq_features = json.load(f)
feat_dim = len(list(iconq_features.values())[0]['features'])
print(f"ICONQ features: {len(iconq_features)} queries, {feat_dim} dims")

# ─── Load trace & build concurrent overlaps ───
trace = []
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f):
        trace.append(row)

timeline = []
for r in trace:
    qid = r['qid']
    runtime = float(r['runtime'])
    status = r['status']
    if status == 'penalty':
        actual_runtime = 60.0
    else:
        actual_runtime = runtime
    start_in_batch = float(r['start'])
    timeline.append((start_in_batch, start_in_batch + actual_runtime,
                     qid, actual_runtime, status))

# Compute concurrent overlaps
concurrent_info = []
for i, (start_i, end_i, qid_i, rt_i, st_i) in enumerate(timeline):
    overlaps = []
    for j, (start_j, end_j, qid_j, _, _) in enumerate(timeline):
        if i == j: continue
        if start_j < end_i and end_j > start_i:
            overlaps.append(qid_j)
    concurrent_info.append((qid_i, start_i, end_i, rt_i, st_i, overlaps))

# Split: 70/30 by time
max_time = max(e for _, _, e, _, _, _ in concurrent_info)
split_time = max_time * 0.7
train_data = [(q, s, e, r, st, ov) for q, s, e, r, st, ov in concurrent_info if s < split_time]
test_data  = [(q, s, e, r, st, ov) for q, s, e, r, st, ov in concurrent_info if s >= split_time]
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ─── Build feature sequences ───
def build_sequences(data_list):
    X, y = [], []
    for qid, start, end, runtime, status, overlaps in data_list:
        if qid not in iconq_features:
            continue
        query_vec = iconq_features[qid]['features'] + [
            math.log(1 + iconq_features[qid]['serial_latency_s'])]
        # query dims = feat_dim + 1 (serial_lat)

        seq = []
        ov_info = []
        for oqid in overlaps:
            if oqid not in iconq_features: continue
            for o_start, _, _, _, _ in [
                (ts, te, q, _, _) for ts, te, q, _, _ in timeline if q == oqid
            ]:
                ov_info.append((o_start, oqid))
                break
        ov_info.sort()

        for o_start, oqid in ov_info:
            ovec = iconq_features[oqid]['features'] + [
                math.log(1 + iconq_features[oqid]['serial_latency_s'])]
            time_diff = start - o_start
            is_before = 1.0 if o_start < start else 0.0
            feat = query_vec + ovec + [time_diff, is_before]
            seq.append(feat)

        if len(seq) > 0:
            X.append(seq)
            y.append(runtime)
    return X, y

X_train, y_train = build_sequences(train_data)
X_test, y_test = build_sequences(test_data)
print(f"Train sequences: {len(X_train)}, Test: {len(X_test)}")

# ─── Pad and normalize ───
def pad_and_normalize(X_list, y_list):
    max_len = max(len(s) for s in X_list)
    d = len(X_list[0][0])
    mask = np.zeros((len(X_list), max_len, d))
    for i, seq in enumerate(X_list):
        mask[i, :len(seq)] = 1.0
        X_list[i] = np.array(seq + [[0.0]*d] * (max_len - len(seq)))

    X_arr = np.array(X_list, dtype=np.float32)
    lengths = np.array([len(s) for s in X_list], dtype=np.int32)

    # Z-score normalize
    X_mean = (X_arr * mask).sum(axis=(0,1)) / max(mask.sum(), 1)
    diff = (X_arr - X_mean) * mask
    X_std = np.sqrt((diff**2).sum(axis=(0,1)) / max(mask.sum(), 1)) + 1e-8

    y_log = np.log(1.0 + np.array(y_list, dtype=np.float32))
    y_mean = y_log.mean()
    y_std = y_log.std() + 1e-8

    return (X_arr, lengths, X_mean, X_std), (y_log, y_mean, y_std)

X_info, y_info = pad_and_normalize(X_train, y_train)
X_arr, lengths, X_mean, X_std = X_info
y_log, y_mean, y_std = y_info
max_len = X_arr.shape[1]

# Test set: use training normalization
def pad_test(X_list):
    max_len = max(len(s) for s in X_list)
    d = len(X_list[0][0])
    X_arr = np.zeros((len(X_list), max_len, d), dtype=np.float32)
    for i, seq in enumerate(X_list):
        X_arr[i, :len(seq)] = seq
    return X_arr, np.array([len(s) for s in X_list], dtype=np.int32)

X_test_arr, test_lengths = pad_test(X_test)
X_test_norm = (X_test_arr - X_mean) / X_std
y_test_log = np.log(1.0 + np.array(y_test, dtype=np.float32))
y_test_norm = (y_test_log - y_mean) / y_std

d = X_arr.shape[2]
print(f"Feature dim: {d}, max_len={max_len}")

# ─── Dataset ───
class SimpleDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.FloatTensor(X)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]

train_ds = SimpleDataset((X_arr - X_mean) / X_std, lengths, (y_log - y_mean) / y_std)
test_ds = SimpleDataset(X_test_norm, test_lengths, y_test_norm)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

# ─── Train ───
model = ConcurrentBiLSTM(input_dim=d)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
best_val, best_state = float('inf'), None

for epoch in range(1, 201):
    model.train()
    for X, lens, y in train_loader:
        opt.zero_grad()
        loss = nn.functional.huber_loss(model(X, lens), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
    scheduler.step()

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X, lens, y in test_loader:
            all_p.append(model(X, lens).numpy())
            all_t.append(y.numpy())
    p = np.concatenate(all_p); t = np.concatenate(all_t)
    p_raw = np.maximum(np.exp(p * y_std + y_mean) - 1, 0.01)
    t_raw = np.maximum(np.exp(t * y_std + y_mean) - 1, 0.01)
    qe = np.maximum(p_raw / t_raw, t_raw / p_raw)
    med = np.median(qe)
    if med < best_val:
        best_val = med
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch % 20 == 0 or epoch == 1:
        print(f'E{epoch:3d} val_med={med:.2f}x best={best_val:.2f}x')

# ─── Final ───
model.load_state_dict(best_state)
model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for X, lens, y in test_loader:
        all_p.append(model(X, lens).numpy())
        all_t.append(y.numpy())
p = np.concatenate(all_p); t = np.concatenate(all_t)
p_raw = np.maximum(np.exp(p * y_std + y_mean) - 1, 0.01)
t_raw = np.maximum(np.exp(t * y_std + y_mean) - 1, 0.01)
qe = np.maximum(p_raw / t_raw, t_raw / p_raw)
qs = np.sort(qe)
nq = len(qs)

ss_r = np.sum((np.log(t_raw + 1) - np.log(p_raw + 1))**2)
ss_t = np.sum((np.log(t_raw + 1) - np.mean(np.log(t_raw + 1)))**2)
r2 = 1 - ss_r / max(ss_t, 1e-8)

print(f"\n=== ICONQ-Style Features ({nq} test queries) ===")
for pct in [50, 80, 90, 95, 99]:
    print(f"  P{pct:2d}: {qs[int(nq*pct/100)]:.2f}x")
print(f"  R²: {r2:.4f}")

# Compare
print(f"\n=== Comparison ===")
print(f"  ICONQ-style (flat {d-2}d):  P50={qs[nq//2]:.2f}x  R²={r2:.4f}")
print(f"  GNN-based   (270d):          P50=1.41x  R²=0.701")
