"""
Ensemble Bi-LSTM training: N seeds, average z-scored predictions.
Uses mixed K=2+3+4 data (5954 train / 982 test, dim=275).
"""

import os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'
MODEL_DIR = os.path.join(DATA_DIR, 'ensemble_models')
os.makedirs(MODEL_DIR, exist_ok=True)


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
        self.y_mean = float(data['y_mean'])
        self.y_std = float(data['y_std'])
        self.X = torch.FloatTensor(X_raw)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor(data['y'].astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.lengths[idx], self.y[idx]


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


class EnsembleBiLSTM(nn.Module):
    """Simple Bi-LSTM regressor with Xavier init and BatchNorm."""
    def __init__(self, input_dim=275, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, 128),
        )
        self.bilstm = nn.LSTM(128, hidden_size, num_layers, dropout=dropout,
                              batch_first=True, bidirectional=True)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
        )
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


def evaluate_raw(model, loader, device='cpu'):
    """Return z-scored predictions and labels."""
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for X, lengths, y in loader:
            X, y = X.to(device), y.to(device)
            all_pred.append(model(X, lengths).cpu().numpy())
            all_label.append(y.cpu().numpy())
    return np.concatenate(all_pred), np.concatenate(all_label)


def train_one(seed, train_loader, test_loader, train_ds, epochs=250, lr=1e-3):
    """Train a single model. Returns best model state and its predictions."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EnsembleBiLSTM(input_dim=train_ds.X.shape[2])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)
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
            p_z, t_z = evaluate_raw(model, test_loader, device)
            p_raw = np.maximum(np.exp(p_z * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
            t_raw = np.maximum(np.exp(t_z * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
            qe = np.maximum(p_raw / t_raw, t_raw / p_raw)
            med = np.median(qe)
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    path = os.path.join(MODEL_DIR, f'model_s{seed}.pt')
    torch.save(best_state, path)

    p_z, t_z = evaluate_raw(model, test_loader, device)
    return p_z, t_z, best_val, n_params


def main():
    train_ds = ConcurrentDataset(os.path.join(DATA_DIR, 'train_data_mixed.npz'))
    test_ds = ConcurrentDataset(os.path.join(DATA_DIR, 'test_data_mixed.npz'))
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Dim: {train_ds.X.shape[2]}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    SEEDS = [42, 123, 456, 789, 1024]
    all_pz, all_tz = [], []
    results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Training seed={seed}...")
        p_z, t_z, best_med, n_params = train_one(seed, train_loader, test_loader, train_ds)
        all_pz.append(p_z)
        all_tz.append(t_z)

        # Single-model metrics
        y_std, y_mean = train_ds.y_std, train_ds.y_mean
        p_raw = np.maximum(np.exp(p_z * y_std + y_mean) - 1, 0.01)
        t_raw = np.maximum(np.exp(t_z * y_std + y_mean) - 1, 0.01)
        qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
        n = len(qe)
        print(f"  Seed {seed}: P50={qe[n//2]:.2f}x P90={qe[int(n*0.9)]:.2f}x P95={qe[int(n*0.95)]:.2f}x  ({n_params:,} params)")
        results.append((seed, qe))

    # Ensemble: average z-scored predictions
    ensemble_pz = np.mean(all_pz, axis=0)
    ensemble_tz = all_tz[0]  # all identical

    y_std, y_mean = train_ds.y_std, train_ds.y_mean
    ep_raw = np.maximum(np.exp(ensemble_pz * y_std + y_mean) - 1, 0.01)
    et_raw = np.maximum(np.exp(ensemble_tz * y_std + y_mean) - 1, 0.01)
    eqe = np.sort(np.maximum(ep_raw / et_raw, et_raw / ep_raw))
    n = len(eqe)

    print(f"\n{'='*60}")
    print(f"Ensemble ({len(SEEDS)} models):")
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f"  P{pct:2d}: {eqe[int(n*pct/100)]:.2f}x")

    # Print comparison table
    print(f"\n{'Model':<20} {'P50':>6} {'P90':>6} {'P95':>6}")
    print('-' * 40)
    for seed, qe in results:
        n = len(qe)
        print(f"  Seed {seed:<13} {qe[n//2]:.2f}x {qe[int(n*0.9)]:.2f}x {qe[int(n*0.95)]:.2f}x")
    print(f"  {'Ensemble':<18} {eqe[n//2]:.2f}x {eqe[int(n*0.9)]:.2f}x {eqe[int(n*0.95)]:.2f}x")

    # R² on log-ratio and absolute runtime
    log_t = ensemble_tz * y_std + y_mean  # denorm to log(1+ratio)
    log_p = ensemble_pz * y_std + y_mean
    r2_ratio = 1 - np.sum((log_t - log_p)**2) / max(np.sum((log_t - log_p.mean())**2), 1e-8)
    print(f"\n  R² (log-ratio): {r2_ratio:.4f}")


if __name__ == '__main__':
    main()
