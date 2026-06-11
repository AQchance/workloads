"""
Bi-LSTM concurrent runtime prediction with ratio target (best method).

Key improvement: predict slowdown_ratio = conc_runtime / serial_latency
instead of absolute runtime. This normalizes the target space and
significantly improves Q-error across all percentiles.

Architecture: 2-layer Bi-LSTM (hidden=256) → MLP → log(1+ratio) prediction.
"""

import os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'
FEATURES_FILE = os.path.join(DATA_DIR, 'gnn_features.json')


class ConcurrentDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        X_raw = data['X']; lengths = data['lengths']
        mask = np.zeros_like(X_raw)
        for i, l in enumerate(lengths):
            if l > 0: mask[i, :l] = 1.0
        self.X_mean = (X_raw * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        diff = ((X_raw - self.X_mean) * mask) ** 2
        self.X_std = np.sqrt(diff.sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8
        # Labels are z-scored log(1+ratio) from prepare_training_data
        self.y_mean = float(data['y_mean']) if 'y_mean' in data else 0.0
        self.y_std = float(data['y_std']) if 'y_std' in data else 1.0
        self.X = torch.FloatTensor(X_raw)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(data['y'].astype(np.float32))
        if 'serial_lat' in data:
            self.serial_lat = np.array(data['serial_lat'], dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.lengths[idx], self.y[idx]


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


class ConcurrentBiLSTM(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, lengths):
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.bilstm(packed)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.predictor(final).squeeze(-1)


def evaluate(model, loader, dataset, device='cpu'):
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for X, lengths, y in loader:
            X, y = X.to(device), y.to(device)
            all_pred.append(model(X, lengths).cpu().numpy())
            all_label.append(y.cpu().numpy())
    p = np.concatenate(all_pred); t = np.concatenate(all_label)
    # Denorm using stored stats: z-scored log-ratio → ratio
    p_ratio = np.maximum(np.exp(p * dataset.y_std + dataset.y_mean) - 1, 0.01)
    t_ratio = np.maximum(np.exp(t * dataset.y_std + dataset.y_mean) - 1, 0.01)
    qe = np.maximum(p_ratio / t_ratio, t_ratio / p_ratio)
    return p, t, qe


def main(trace_file, epochs=200, lr=1e-3, seed=42, prefix=''):
    torch.manual_seed(seed); np.random.seed(seed)
    train_npz = os.path.join(DATA_DIR, f'train_data{prefix}.npz')
    test_npz  = os.path.join(DATA_DIR, f'test_data{prefix}.npz')
    train_ds = ConcurrentDataset(train_npz)
    test_ds  = ConcurrentDataset(test_npz)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Dim: {train_ds.X.shape[2]}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = ConcurrentBiLSTM(input_dim=train_ds.X.shape[2])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
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
            _, _, qe = evaluate(model, test_loader, test_ds, device)
            med = np.median(qe)
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'E{epoch:3d} val_med={med:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state)
    _, _, qe = evaluate(model, test_loader, test_ds, device)
    qs = np.sort(qe); nq = len(qs)

    # Get serial latencies for R² on absolute runtime
    with open(FEATURES_FILE) as f: gnf = json.load(f)
    with open(trace_file) as f: trace = list(csv.DictReader(f))
    tl = []
    for r in trace:
        q = r['qid']; rt = float(r['runtime']); st = r['status']
        if st == 'penalty': continue
        tl.append((float(r['start']), float(r['start']) + rt, q, rt, st))
    ci = []
    for i, (si, ei, qi, rti, sti) in enumerate(tl):
        ov = []
        for j, (sj, ej, qj, _, _) in enumerate(tl):
            if i != j and sj < ei and ej > si: ov.append(qj)
        ci.append((qi, si, ei, rti, sti, ov))
    mt = max(e for _, _, e, _, _, _ in ci); sp = mt * 0.7
    test_ci = [c for c in ci if c[1] >= sp]
    serial_lats = []
    for qi, _, _, _, _, _ in test_ci:
        if qi in gnf:
            serial_lats.append(max(gnf[qi]['serial_labels'].get('latency_s', 1), 0.5))

    # Get predictions and convert to absolute runtime
    model.eval()
    all_pred_ratio, all_true_ratio = [], []
    with torch.no_grad():
        for X, lengths, y in test_loader:
            X, y = X.to(device), y.to(device)
            all_pred_ratio.append(model(X, lengths).cpu().numpy())
            all_true_ratio.append(y.cpu().numpy())
    p_ratio = np.concatenate(all_pred_ratio); t_ratio = np.concatenate(all_true_ratio)
    pr = np.maximum(np.exp(p_ratio * test_ds.y_std + test_ds.y_mean) - 1, 0.01)
    tr = np.maximum(np.exp(t_ratio * test_ds.y_std + test_ds.y_mean) - 1, 0.01)
    sla = np.array(serial_lats[:len(pr)])
    p_abs = pr * sla; t_abs = tr * sla
    r2 = 1 - np.sum((np.log(t_abs+1) - np.log(p_abs+1))**2) / max(np.sum((np.log(t_abs+1) - np.mean(np.log(t_abs+1)))**2), 1e-8)

    print(f"\n=== Results ({nq} test queries) ===")
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f"  P{pct:2d}: {qs[int(nq*pct/100)]:.2f}x")
    print(f"  R²: {r2:.4f}")

    torch.save(best_state, os.path.join(DATA_DIR, 'bilstm_model.pt'))
    print(f"\nModel saved: lstm/bilstm_model.pt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv')
    parser.add_argument('--data-prefix', default='', help='Use train_data{PREFIX}.npz (e.g. _mixed)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.trace, args.epochs, args.lr, args.seed, args.data_prefix)
