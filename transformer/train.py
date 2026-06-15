"""
Train Concurrent Query Transformer with resource-aware attention.
"""

import os, sys, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ConcurrentQueryTransformer

DATA_DIR = '/home/anqian/code/python/workloads/transformer'


class TransformerDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.nodes = torch.FloatTensor(d['nodes'])
        self.pairs = torch.FloatTensor(d['pairs'])
        self.times = torch.FloatTensor(d['times'])
        self.resources = torch.FloatTensor(d['resources'])
        self.y = torch.FloatTensor(d['y'])
        self.y_mean = float(d['y_mean'])
        self.y_std = float(d['y_std'])
        self.serial_lat = np.array(d['serial_lat'], dtype=np.float32)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return (self.nodes[idx], self.pairs[idx], self.times[idx],
                self.resources[idx], self.y[idx],
                torch.tensor(self.serial_lat[idx], dtype=torch.float32))


def main(epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    train_ds = TransformerDataset(os.path.join(DATA_DIR, 'train_transformer.npz'))
    test_ds = TransformerDataset(os.path.join(DATA_DIR, 'test_transformer.npz'))
    print(f'Train: {len(train_ds)} | Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = ConcurrentQueryTransformer().cuda()
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0; n_batches = 0
        for nodes, pairs, times, res, y, _ in train_loader:
            nodes, pairs, times = nodes.cuda(), pairs.cuda(), times.cuda()
            res, y = res.cuda(), y.cuda()
            opt.zero_grad()
            # Target is always index 0
            target_mask = torch.zeros(nodes.shape[0], nodes.shape[1], dtype=torch.bool)
            target_mask[:, 0] = True
            target_mask = target_mask.cuda()
            pred = model(nodes, pairs, times, res, target_mask)
            loss = nn.functional.huber_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_loss += loss.item(); n_batches += 1
        scheduler.step()

        # Evaluate
        model.eval()
        all_pred, all_true, all_sl = [], [], []
        with torch.no_grad():
            for nodes, pairs, times, res, y, sl in test_loader:
                nodes, pairs, times = nodes.cuda(), pairs.cuda(), times.cuda()
                res, y = res.cuda(), y.cuda()
                target_mask = torch.zeros(nodes.shape[0], nodes.shape[1], dtype=torch.bool)
                target_mask[:, 0] = True; target_mask = target_mask.cuda()
                pred = model(nodes, pairs, times, res, target_mask)
                all_pred.append(pred.cpu().numpy())
                all_true.append(y.cpu().numpy())
                all_sl.append(sl.numpy())

        p = np.concatenate(all_pred); t = np.concatenate(all_true)
        pr = np.maximum(np.exp(p * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
        tr = np.maximum(np.exp(t * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
        qe = np.maximum(pr / tr, tr / pr)
        med = np.median(qe)

        if med < best_val:
            best_val = med
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            print(f'E{epoch:3d} train={train_loss/n_batches:.4f} val_med={med:.2f}x best={best_val:.2f}x')

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for nodes, pairs, times, res, y, _ in test_loader:
            nodes, pairs, times = nodes.cuda(), pairs.cuda(), times.cuda()
            res, y = res.cuda(), y.cuda()
            target_mask = torch.zeros(nodes.shape[0], nodes.shape[1], dtype=torch.bool)
            target_mask[:, 0] = True; target_mask = target_mask.cuda()
            pred = model(nodes, pairs, times, res, target_mask)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

    p = np.concatenate(all_pred); t = np.concatenate(all_true)
    pr = np.maximum(np.exp(p * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
    tr = np.maximum(np.exp(t * train_ds.y_std + train_ds.y_mean) - 1, 0.01)
    qe = np.sort(np.maximum(pr / tr, tr / pr))
    n = len(qe)

    print(f'\n=== Transformer Results ({n} queries) ===')
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f'  P{pct:2d}: {qe[int(n * pct / 100)]:.2f}x')

    torch.save(best_state, os.path.join(DATA_DIR, 'transformer_best.pt'))
    print(f'\nModel saved: transformer/transformer_best.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.epochs, args.lr, args.seed)
