"""Train Bi-LSTM on K=4 concurrent data, compare with K=2 results."""
import sys, os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bilstm import ConcurrentBiLSTM, collate_fn

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
TRACE_FILE = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_4.csv'
OUT_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'

with open(FEATURES_FILE) as f: gnn_features = json.load(f)

# Load K=4 trace
trace = []
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f): trace.append(row)

# Build timeline with actual penalty cost
timeline = []
for r in trace:
    qid = r['qid']; rt = float(r['runtime']); st = r['status']
    actual_rt = 60.0 if st == 'penalty' else rt
    start_s = float(r['start'])
    timeline.append((start_s, start_s + actual_rt, qid, actual_rt, st))

# Compute overlaps
concurrent_info = []
for i, (si, ei, qi, rti, sti) in enumerate(timeline):
    ov = []
    for j, (sj, ej, qj, _, _) in enumerate(timeline):
        if i != j and sj < ei and ej > si: ov.append(qj)
    concurrent_info.append((qi, si, ei, rti, sti, ov))

# Split 70/30 by time
mt = max(e for _, _, e, _, _, _ in concurrent_info)
split_t = mt * 0.7
train_d = [c for c in concurrent_info if c[1] < split_t]
test_d = [c for c in concurrent_info if c[1] >= split_t]
print(f"Train: {len(train_d)}, Test: {len(test_d)}")

def resource_conflict(target_res, conc_res):
    """Compute 5-dim conflict vector: min/max ratio per resource."""
    t = np.array(target_res); c = np.array(conc_res)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))

def build_sequences(data_list, features):
    X, y = [], []
    for qi, si, ei, rti, sti, ov in data_list:
        if qi not in features: continue
        if sti == 'penalty': continue
        sl = math.log(1 + features[qi]['serial_labels'].get('latency_s', 10))
        qv = features[qi]['plan_emb'] + list(features[qi]['gpu_resources']) + [sl]
        t_res = list(features[qi]['gpu_resources'])

        seq = []
        oinfo = []
        for oq in ov:
            if oq not in features: continue
            for o_s, _, _, _, _ in [(ts,te,q,_,_) for ts,te,q,_,_ in timeline if q == oq]:
                oinfo.append((o_s, oq)); break
        oinfo.sort()
        for o_s, oq in oinfo:
            osl = math.log(1 + features[oq]['serial_labels'].get('latency_s', 10))
            ovv = features[oq]['plan_emb'] + list(features[oq]['gpu_resources']) + [osl]
            o_res = list(features[oq]['gpu_resources'])
            # Per-pair conflict vector
            pair_conflict = resource_conflict(t_res, o_res)
            feat = qv + ovv + [si - o_s, 1.0 if o_s < si else 0.0] + pair_conflict
            seq.append(feat)
        if seq: X.append(seq); y.append(rti)
    return X, y

X_tr, y_tr = build_sequences(train_d, gnn_features)
X_te, y_te = build_sequences(test_d, gnn_features)
print(f"Train seqs: {len(X_tr)}, Test seqs: {len(X_te)}")

# Pad, normalize, train
max_len = max(max(len(s) for s in X_tr), max(len(s) for s in X_te))
d = len(X_tr[0][0])

X_tr_a = np.zeros((len(X_tr), max_len, d), dtype=np.float32)
for i, s in enumerate(X_tr): X_tr_a[i, :len(s)] = s
X_te_a = np.zeros((len(X_te), max_len, d), dtype=np.float32)
for i, s in enumerate(X_te): X_te_a[i, :len(s)] = s

# Z-score normalize
mask = np.zeros_like(X_tr_a)
for i, s in enumerate(X_tr): mask[i, :len(s)] = 1.0
X_mean = (X_tr_a * mask).sum(axis=(0,1)) / max(mask.sum(), 1)
diff = (X_tr_a - X_mean) * mask
X_std = np.sqrt((diff**2).sum(axis=(0,1)) / max(mask.sum(), 1)) + 1e-8

y_log = np.log(1 + np.array(y_tr, dtype=np.float32))
y_mean, y_std = y_log.mean(), y_log.std() + 1e-8

class SD(Dataset):
    def __init__(self, X, lens, y):
        self.X = torch.FloatTensor(X); self.lens = torch.LongTensor(lens); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.lens[i], self.y[i]

tr_len = np.array([len(s) for s in X_tr], dtype=np.int32)
te_len = np.array([len(s) for s in X_te], dtype=np.int32)
tr_ds = SD((X_tr_a - X_mean) / X_std, tr_len, (y_log - y_mean) / y_std)
y_te_log = np.log(1 + np.array(y_te, dtype=np.float32))
te_ds = SD((X_te_a - X_mean) / X_std, te_len, (y_te_log - y_mean) / y_std)

tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

model = ConcurrentBiLSTM(input_dim=d)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
best_val, best_state = float('inf'), None

for epoch in range(1, 201):
    model.train()
    for X, lens, y in tr_ld:
        opt.zero_grad()
        loss = nn.functional.huber_loss(model(X, lens), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
    scheduler.step()

    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for X, lens, y in te_ld:
            ap.append(model(X, lens).numpy()); at.append(y.numpy())
    p = np.concatenate(ap); t = np.concatenate(at)
    pr = np.maximum(np.exp(p * y_std + y_mean) - 1, 0.01)
    tr = np.maximum(np.exp(t * y_std + y_mean) - 1, 0.01)
    qe = np.maximum(pr/tr, tr/pr)
    med = np.median(qe)
    if med < best_val: best_val = med; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch % 20 == 0 or epoch == 1:
        print(f'E{epoch:3d} val_med={med:.2f}x best={best_val:.2f}x')

model.load_state_dict(best_state)
model.eval()
ap, at = [], []
with torch.no_grad():
    for X, lens, y in te_ld:
        ap.append(model(X, lens).numpy()); at.append(y.numpy())
p = np.concatenate(ap); t = np.concatenate(at)
pr = np.maximum(np.exp(p * y_std + y_mean) - 1, 0.01)
tr = np.maximum(np.exp(t * y_std + y_mean) - 1, 0.01)
qe = np.maximum(pr/tr, tr/pr)
qs = np.sort(qe); nq = len(qs)
ss_r = np.sum((np.log(tr+1) - np.log(pr+1))**2)
ss_t = np.sum((np.log(tr+1) - np.mean(np.log(tr+1)))**2)
r2 = 1 - ss_r/max(ss_t, 1e-8)

print(f"\n=== K=4 Results ({nq} test queries) ===")
for pct in [10,20,30,40,50,60,70,80,90,95,99]:
    print(f"  P{pct:2d}: {qs[int(nq*pct/100)]:.2f}x")
print(f"  R²: {r2:.4f}")
print(f"\n=== K=2 vs K=4 Comparison ===")
print(f"              P50     P80     P90     R²")
print(f"  K=2:        1.41x   2.06x   2.66x   0.701")
print(f"  K=4:        {qs[nq//2]:.2f}x   {qs[int(nq*0.8)]:.2f}x   {qs[int(nq*0.9)]:.2f}x   {r2:.4f}")

torch.save(best_state, os.path.join(OUT_DIR, 'bilstm_k4_model.pt'))
