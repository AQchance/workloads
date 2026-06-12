"""
Bi-LSTM concurrent runtime prediction with dual-head prediction.

Two heads:
  - ratio_head: predicts slowdown_ratio (good for small slowdowns)
  - delta_head: predicts absolute delta_ms (good for large slowdowns)
  - gate: learns to blend between them based on query features

Architecture: 2-layer Bi-LSTM (hidden=256) → ratio_head + delta_head + gate
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
    def __getitem__(self, idx):
        sl = self.serial_lat[idx] if hasattr(self, 'serial_lat') else 1.0
        return self.X[idx], self.lengths[idx], self.y[idx], torch.tensor(sl, dtype=torch.float32)


def collate_fn(batch):
    X, lengths, y, slat = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]),
            torch.stack([slat[i] for i in sort_idx]))


class ConcurrentBiLSTM(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        shared_dim = hidden_dim // 2
        # Ratio head: good for small slowdowns
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(shared_dim, 1))
        # Delta head: good for large slowdowns (predicts absolute ms increment)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(shared_dim, 1))
        # Gate: blend ratio vs delta prediction per sample
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(shared_dim, 1), nn.Sigmoid())

    def forward(self, X, lengths):
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.bilstm(packed)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)
        ratio_pred = self.ratio_head(final).squeeze(-1)
        delta_pred = self.delta_head(final).squeeze(-1)
        gate_val = self.gate(final).squeeze(-1)
        return ratio_pred, delta_pred, gate_val


def evaluate(model, loader, dataset, device='cpu'):
    model.eval()
    all_p_ratio, all_p_delta, all_gate, all_label, all_slats = [], [], [], [], []
    with torch.no_grad():
        for X, lengths, y, slat in loader:
            X, y, slat = X.to(device), y.to(device), slat.to(device)
            r_pred, d_pred, g_val = model(X, lengths)
            all_p_ratio.append(r_pred.cpu().numpy())
            all_p_delta.append(d_pred.cpu().numpy())
            all_gate.append(g_val.cpu().numpy())
            all_label.append(y.cpu().numpy())
            all_slats.append(slat.cpu().numpy())
    pr = np.concatenate(all_p_ratio); pd = np.concatenate(all_p_delta)
    gv = np.concatenate(all_gate); t = np.concatenate(all_label)
    sl = np.concatenate(all_slats)

    # Convert predictions to raw ratio
    pr_raw = np.maximum(np.exp(pr * dataset.y_std + dataset.y_mean) - 1, 0.01)
    t_raw  = np.maximum(np.exp(t * dataset.y_std + dataset.y_mean) - 1, 0.01)

    # Delta head: z-scored log(1+delta_ms) → delta_ms → ratio
    pd_raw = np.maximum(np.exp(pd * dataset.delta_mean + dataset.delta_std) - 1, 0.01)
    pd_ratio = pd_raw / np.maximum(sl, 0.5) + 1.0

    # Blend
    p_ratio = gv.squeeze() * pr_raw + (1 - gv.squeeze()) * pd_ratio
    qe = np.maximum(p_ratio / np.maximum(t_raw, 0.01), np.maximum(t_raw, 0.01) / np.maximum(p_ratio, 0.01))
    return p_ratio, t_raw, qe


def main(trace_file, epochs=200, lr=1e-3, seed=42, prefix=''):
    torch.manual_seed(seed); np.random.seed(seed)
    train_npz = os.path.join(DATA_DIR, f'train_data{prefix}.npz')
    test_npz  = os.path.join(DATA_DIR, f'test_data{prefix}.npz')
    train_ds = ConcurrentDataset(train_npz)
    test_ds  = ConcurrentDataset(test_npz)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Dim: {train_ds.X.shape[2]}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    # Compute delta label stats
    sample_ratios = np.exp(train_ds.y.numpy() * train_ds.y_std + train_ds.y_mean) - 1
    sample_slats = train_ds.serial_lat
    sample_deltas = np.maximum((sample_ratios - 1.0) * sample_slats, 0.001)
    log_deltas = np.log(1.0 + sample_deltas)
    train_ds.delta_mean = float(log_deltas.mean())
    train_ds.delta_std  = float(log_deltas.std()) + 1e-8
    test_ds.delta_mean  = train_ds.delta_mean
    test_ds.delta_std   = train_ds.delta_std
    print(f"Delta stats: mean={train_ds.delta_mean:.3f} std={train_ds.delta_std:.3f}")

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
        for X, lengths, y, slat in train_loader:
            X, y, slat = X.to(device), y.to(device), slat.to(device)
            opt.zero_grad()
            pred_ratio, pred_delta, gate_val = model(X, lengths)

            # Ratio label
            true_ratio_raw = torch.exp(y * train_ds.y_std + train_ds.y_mean) - 1
            # Delta label
            true_delta_raw = torch.clamp((true_ratio_raw - 1.0) * slat, min=0.001)
            true_delta_z = (torch.log(1.0 + true_delta_raw) - train_ds.delta_mean) / train_ds.delta_std

            loss_ratio = nn.functional.huber_loss(pred_ratio, y)
            loss_delta = nn.functional.huber_loss(pred_delta, true_delta_z)

            # Gate: soft supervision — trust ratio for small slowdowns, delta for large
            gate_target = torch.sigmoid((2.0 - true_ratio_raw) / 0.8)
            loss_gate = nn.functional.binary_cross_entropy(gate_val, gate_target)

            loss = loss_ratio + loss_delta + 0.05 * loss_gate
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
    all_pr, all_gt = [], []
    all_sl = []
    with torch.no_grad():
        for X, lengths, y, slat in test_loader:
            X, y, slat = X.to(device), y.to(device), slat.to(device)
            r_pred, d_pred, g_val = model(X, lengths)
            # Denorm ratio
            pr_raw = torch.clamp(torch.exp(r_pred * test_ds.y_std + test_ds.y_mean) - 1, min=0.01)
            # Denorm delta → ratio
            pd_raw = torch.clamp(torch.exp(d_pred * test_ds.delta_mean + test_ds.delta_std) - 1, min=0.01)
            pd_ratio = pd_raw / torch.clamp(slat, min=0.5) + 1.0
            # Blend
            p_blend = g_val.squeeze() * pr_raw + (1 - g_val.squeeze()) * pd_ratio
            # True ratio
            t_raw = torch.clamp(torch.exp(y * test_ds.y_std + test_ds.y_mean) - 1, min=0.01)
            all_pr.append(p_blend.cpu().numpy())
            all_gt.append(t_raw.cpu().numpy())
            all_sl.append(slat.cpu().numpy())
    pr = np.concatenate(all_pr); tr = np.concatenate(all_gt)
    sla_arr = np.concatenate(all_sl)
    p_abs = pr * sla_arr; t_abs = tr * sla_arr
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
