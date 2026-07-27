"""
Train Bi-LSTM on ALL 3 rounds of Poisson traces (final deployment model).
Saves model + normalization for ICONQ scheduler.
"""
import os, csv, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DIR = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent'
RESOURCE_PATH = f'{DIR}/tabpfn_258_predictions_oof.json'

DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']
SEED = 42
EPOCHS = 300


def resource_conflict(t, c):
    t, c = np.array(t), np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


def build_sequences_from_events(ci, cache):
    qid_info = {}
    for qi, si, ei, _, _, _ in ci:
        qid_info[qi] = (si, ei)
    X, y_ratio = [], []
    for qi, si, ei, rti, sti, ov in ci:
        if sti == 'penalty' or qi not in cache:
            continue
        pred_i = cache[qi]
        serial_lat = max(pred_i.get('serial_lat_s', 10), 0.5)
        qv = [pred_i[d] for d in DIMS]
        tr_ = [pred_i[d] for d in DIMS]
        seq = []
        peers = [(qid_info[oq][0], oq) for oq in ov if oq in qid_info and oq in cache]
        peers.sort()
        for osv, oq in peers:
            pred_j = cache[oq]
            ovv = [pred_j[d] for d in DIMS]
            oc = [pred_j[d] for d in DIMS]
            c = resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - osv, 1.0 if osv < si else 0.0] + c)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)
    return X, y_ratio


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
    return (torch.stack([X[i] for i in si]),
            torch.stack([l[i] for i in si]),
            torch.stack([y[i] for i in si]))


class BiLSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


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


def main():
    print('=' * 60)
    print('Bi-LSTM Final Model: Train on ALL 3 Rounds')
    print('=' * 60)

    with open(RESOURCE_PATH) as f:
        cache = json.load(f)
    print(f'[1] TabPFN resources: {len(cache)} queries')

    # Load arrival rounds map
    round_map = {}
    with open(f'{DIR}/arrival_times_3r_poisson.csv') as f:
        for row in csv.DictReader(f):
            round_map[(row['qid'], round(float(row['arrival_time_s']), 3))] = int(row['round'])

    # Load all traces
    ci_all = []
    for trace_name in ['fifo_k4_trace.csv', 'fifo_k8_trace.csv']:
        path = f'{DIR}/{trace_name}'
        trace = []
        with open(path) as f:
            for row in csv.DictReader(f):
                t0 = float(row['start'])
                rt = float(row['runtime'])
                trace.append((row['qid'], t0, t0 + rt, rt, row['status'],
                              round(float(row['arrival']), 3)))
        for i, (qi, si, ei, rti, sti, arr) in enumerate(trace):
            ov = [qj for j, (qj, sj, ej, _, _, _) in enumerate(trace)
                  if i != j and sj < ei and ej > si]
            rnd = round_map.get((qi, arr), 0)
            ci_all.append((qi, si, ei, rti, sti, ov, rnd))

    from collections import Counter
    rc = Counter(c[6] for c in ci_all)
    print(f'[2] Events per round: {dict(sorted(rc.items()))}')
    print(f'    Total: {len(ci_all)} events')

    # Use all data (rounds 1,2,3)
    ci_all_noround = [c[:6] for c in ci_all]
    X_all, y_all = build_sequences_from_events(ci_all_noround, cache)
    print(f'[3] Total sequences: {len(X_all)}')

    d_in = len(X_all[0][0]) if X_all else 17
    print(f'[4] Input dim: {d_in}')

    ml = max(len(s) for s in X_all)
    Xn, l_n, yn, Xm, Xs, ym, ys = pad_normalize(X_all, y_all, ml)

    # 90/10 train/val split for early stopping
    np.random.seed(SEED)
    idx = np.random.permutation(len(Xn))
    split_v = int(len(Xn) * 0.9)
    train_ds = RatioDataset(Xn[idx[:split_v]], l_n[idx[:split_v]], yn[idx[:split_v]])
    val_ds = RatioDataset(Xn[idx[split_v:]], l_n[idx[split_v:]], yn[idx[split_v:]])

    print(f'[5] Train: {len(train_ds)}, Val: {len(val_ds)}')

    # Training
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = BiLSTM(input_dim=d_in).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
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

        if epoch % 30 == 0 or epoch == 1 or epoch == EPOCHS:
            model.eval()
            ap, at = [], []
            with torch.no_grad():
                for X, l, y in val_loader:
                    ap.append(model(X.to(device), l).cpu().numpy())
                    at.append(y.numpy())
            pz, tz = np.concatenate(ap), np.concatenate(at)
            pr = np.maximum(np.exp(pz * ys + ym) - 1, 0.01)
            tr = np.maximum(np.exp(tz * ys + ym) - 1, 0.01)
            med = np.median(np.maximum(pr / tr, tr / pr))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'  E{epoch:3d} val={med:.2f}x best={best_val:.2f}x ({n_params:,} params)')

    model.load_state_dict(best_state)
    print(f'\n[6] Best val Q-error: {best_val:.2f}x')

    # Save model (compatible format with scheduler)
    model_path = f'{DIR}/bilstm_3r.pt'
    norm_path = f'{DIR}/bilstm_3r_norm.npz'
    torch.save(best_state, model_path)
    np.savez(norm_path, X_mean=Xm, X_std=Xs, y_mean=ym, y_std=ys)
    print(f'[7] Saved: {model_path}')
    print(f'     Norm:  {norm_path}')
    print(f'     X_mean shape={Xm.shape}, y_mean={ym:.4f}, y_std={ys:.4f}')


if __name__ == '__main__':
    main()
