"""
Step 3: Train Bi-LSTM for concurrent runtime prediction.

Architecture: 2-layer Bi-LSTM (hidden=256) → MLP → 1 scalar
Training in log(1+x) z-score space (same as GNN).
"""

import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/home/anqian/Desktop/my_lab/workloads/lstm'


class ConcurrentDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        X_raw = data['X']
        lengths = data['lengths']

        # Z-score normalize features per dimension
        mask = np.zeros_like(X_raw)
        for i, l in enumerate(lengths):
            if l > 0: mask[i, :l] = 1.0
        self.X_mean = (X_raw * mask).sum(axis=(0, 1)) / max(mask.sum(), 1)
        diff = (X_raw - self.X_mean) * mask
        self.X_std = np.sqrt((diff ** 2).sum(axis=(0, 1)) / max(mask.sum(), 1)) + 1e-8

        # Log-transform labels, z-score
        y_log = np.log(1.0 + data['y'])
        self.y_mean = y_log.mean()
        self.y_std = y_log.std() + 1e-8

        self.X = torch.FloatTensor((X_raw - self.X_mean) / self.X_std)
        self.lengths = torch.LongTensor(lengths)
        self.y = torch.FloatTensor((y_log - self.y_mean) / self.y_std)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]


def collate_fn(batch):
    X, lengths, y = zip(*batch)
    sort_idx = torch.argsort(torch.stack(lengths), descending=True)
    X = torch.stack([X[i] for i in sort_idx])
    lengths = torch.stack([lengths[i] for i in sort_idx])
    y = torch.stack([y[i] for i in sort_idx])
    return X, lengths, y


class ConcurrentBiLSTM(nn.Module):
    def __init__(self, input_dim=268, hidden_dim=256, num_layers=2, dropout=0.2):
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


def evaluate(model, loader, dataset):
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for X, lengths, y in loader:
            pred = model(X, lengths)
            all_pred.append(pred.numpy())
            all_label.append(y.numpy())
    p = np.concatenate(all_pred)
    t = np.concatenate(all_label)
    # Denormalize to raw seconds
    p_raw = np.exp(p * dataset.y_std + dataset.y_mean) - 1
    t_raw = np.exp(t * dataset.y_std + dataset.y_mean) - 1
    p_raw = np.maximum(p_raw, 0.01)
    t_raw = np.maximum(t_raw, 0.01)
    qe = np.maximum(p_raw / t_raw, t_raw / p_raw)
    return p, t, qe, p_raw, t_raw


def main():
    train_ds = ConcurrentDataset(os.path.join(DATA_DIR, 'train_data.npz'))
    test_ds = ConcurrentDataset(os.path.join(DATA_DIR, 'test_data.npz'))
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Dim: {train_ds.X.shape[2]}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = ConcurrentBiLSTM(input_dim=train_ds.X.shape[2])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)

    best_val, best_state = float('inf'), None

    for epoch in range(1, 201):
        model.train()
        train_loss = 0.0
        for X, lengths, y in train_loader:
            opt.zero_grad()
            pred = model(X, lengths)
            loss = nn.functional.huber_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_loss += loss.item()
        scheduler.step()

        _, _, qe, _, _ = evaluate(model, test_loader, test_ds)
        med_qe = np.median(qe)

        if med_qe < best_val:
            best_val = med_qe
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f'E{epoch:3d} loss={train_loss/len(train_loader):.4f} '
                  f'val_med={med_qe:.2f}x best={best_val:.2f}x')

    model.load_state_dict(best_state)
    _, _, qe, p_raw, t_raw = evaluate(model, test_loader, test_ds)
    qs = np.sort(qe)
    ss_r = np.sum((np.log(t_raw + 1) - np.log(p_raw + 1)) ** 2)
    ss_t = np.sum((np.log(t_raw + 1) - np.mean(np.log(t_raw + 1))) ** 2)
    r2 = 1 - ss_r / max(ss_t, 1e-8)

    print(f"\n=== Final ({len(qs)} queries) ===")
    for pct in [50, 80, 90, 95, 99]:
        print(f"  P{pct:2d}: {qs[int(len(qs)*pct/100)]:.2f}x")
    print(f"  R²: {r2:.4f}")
    print(f"  Predict mean baseline: P50={np.median(np.maximum(np.mean(t_raw)/t_raw, t_raw/np.mean(t_raw))):.2f}x")

    torch.save(best_state, os.path.join(DATA_DIR, 'bilstm_model.pt'))


if __name__ == '__main__':
    main()
