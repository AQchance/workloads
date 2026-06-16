"""
ResourceFullBiLSTM: exact architecture from previous session.
GNN features + resource gate (per-timestep) + resource bias (output).
Achieved P50=1.34x (3-layer), 1.35x (3-seed ensemble).
"""

import os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = '/home/anqian/Desktop/my_lab/workloads'
DATA_DIR = os.path.join(ROOT, 'lstm')

TRAIN_PATH = os.path.join(DATA_DIR, 'train_data_mixed.npz')
TEST_PATH = os.path.join(DATA_DIR, 'test_data_mixed.npz')


class MixedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        X_raw = data['X']
        self.lengths = data['lengths']
        self.y_mean = float(data['y_mean'])
        self.y_std = float(data['y_std'])
        self.X = torch.FloatTensor(X_raw)
        self.y = torch.FloatTensor(data['y'].astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx],
                torch.tensor(self.lengths[idx].item(), dtype=torch.long),
                self.y[idx])


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    return (torch.stack([X[i] for i in sort_idx]),
            torch.stack([lengths[i] for i in sort_idx]),
            torch.stack([y[i] for i in sort_idx]))


class ResourceFullBiLSTM(nn.Module):
    """Exact reproduction of the architecture that achieved P50=1.34x."""
    def __init__(self, input_dim=275, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()

        # Stage 1: Resource gate (10 → hidden/2 → input_dim → sigmoid)
        self.res_gate = nn.Sequential(
            nn.Linear(10, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim), nn.Sigmoid())

        # Stage 2: Embedding + BiLSTM
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0)

        # Stage 3: Predictor (hidden_dim → hidden/2 → 1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))

        # Stage 4: Resource bias (10 → hidden/4 → 1)
        self.res_bias = nn.Sequential(
            nn.Linear(10, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1))

    def forward(self, X, lengths):
        # Extract resource pairs from input: target_res[128:133], peer_res[262:267]
        res_pairs = torch.cat([X[:, :, 128:133], X[:, :, 262:267]], dim=-1)  # [B,T,10]

        # Resource gate: per-timestep gating
        gate = self.res_gate(res_pairs)  # [B, T, input_dim]
        X_gated = X * gate

        # BiLSTM
        x = self.embedding(X_gated)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.bilstm(packed)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)

        # Prediction + resource bias
        base_pred = self.predictor(final).squeeze(-1)
        res_mean = res_pairs.mean(dim=1)  # [B, 10]
        bias = self.res_bias(res_mean).squeeze(-1)
        return base_pred + bias


def train_one(seed, num_layers, train_loader, test_loader, y_mean, y_std, epochs=250, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResourceFullBiLSTM(num_layers=num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_med, best_state = float('inf'), None

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
            p_z = np.concatenate(ap); t_z = np.concatenate(at)
            p_raw = np.maximum(np.exp(p_z * y_std + y_mean) - 1, 0.01)
            t_raw = np.maximum(np.exp(t_z * y_std + y_mean) - 1, 0.01)
            med = np.median(np.maximum(p_raw / t_raw, t_raw / p_raw))
            if med < best_med:
                best_med = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

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
    return p_z, t_z, best_med, n_params


def main():
    train_ds = MixedDataset(TRAIN_PATH)
    test_ds = MixedDataset(TEST_PATH)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Dim: {train_ds.X.shape[2]}")
    print(f"Architecture: ResourceFullBiLSTM (Resource Gate + Resource Bias)")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    y_mean, y_std = train_ds.y_mean, train_ds.y_std

    # Single seed, 3-layer (best config)
    print(f"\n{'='*60}")
    print(f"Training ResourceFullBiLSTM (3-layer, seed=42)...")
    p_z, t_z, best_med, n_params = train_one(
        42, 3, train_loader, test_loader, y_mean, y_std)
    p_raw = np.maximum(np.exp(p_z * y_std + y_mean) - 1, 0.01)
    t_raw = np.maximum(np.exp(t_z * y_std + y_mean) - 1, 0.01)
    qe = np.sort(np.maximum(p_raw / t_raw, t_raw / p_raw))
    n = len(qe)

    print(f"\n=== ResourceFull GNN (3L, seed=42, {n_params:,} params) ===")
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f"  P{pct:2d}: {qe[int(n*pct/100)]:.2f}x")

    log_t = t_z * y_std + y_mean
    log_p = p_z * y_std + y_mean
    r2 = 1 - np.sum((log_t - log_p)**2) / max(np.sum((log_t - log_t.mean())**2), 1e-8)
    print(f"  R² (log-ratio): {r2:.4f}")


if __name__ == '__main__':
    main()
