"""
Train PlanGNN with cgroup-measured physical resource labels.
Uses native PlanGNN (model.py) without monkey-patches.
"""

import sys, os, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PlanGNN


def load_cgroup_labels(cgroup_dir: str) -> dict:
    labels = {}
    for f in os.listdir(cgroup_dir):
        if not f.endswith('.json') or f == 'summary.csv':
            continue
        with open(os.path.join(cgroup_dir, f)) as fp:
            data = json.load(fp)
        if data.get('status') != 'ok':
            continue
        labels[data['qid']] = {
            'memory_bytes': sum(data['memory_delta_bytes'].values()),
            'network_bytes': sum(data['network_delta_bytes'].values()),
            'disk_bytes': sum(data['disk_delta_bytes'].values()),
            'latency_s': data['latency_s'],
        }
    return labels


def load_dataset(plan_dir, ndv_cache, dist_cache, cgroup_labels):
    from train_ndv import parse_plan
    graphs, labels_out, meta = [], [], []
    for qid, clab in cgroup_labels.items():
        pf = os.path.join(plan_dir, f'{qid}.txt')
        if not os.path.exists(pf): continue
        with open(pf) as f: plan_text = f.read()
        g = parse_plan(plan_text, ndv_cache, dist_cache)
        if g is None or g.x.shape[0] == 0: continue
        graphs.append(g); labels_out.append(clab); meta.append(qid)
    return graphs, labels_out, meta


def normalize_labels(labels):
    keys = ['memory_bytes', 'network_bytes', 'disk_bytes', 'latency_s']
    log_labels = [{k: math.log(1.0 + max(l[k], 1)) for k in keys} for l in labels]
    stats = {}
    for k in keys:
        vals = [l[k] for l in log_labels]
        stats[k] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-8)}
    return [{k: (l[k] - stats[k]["mean"]) / stats[k]["std"] for k in keys} for l in log_labels], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-dir', default='/home/anqian/Desktop/my_lab/workloads/explain_plans')
    parser.add_argument('--cgroup-dir', default='/home/anqian/Desktop/my_lab/workloads/cgroup_resources')
    parser.add_argument('--ndv-cache', default='/home/anqian/Desktop/my_lab/workloads/ndv_cache.json')
    parser.add_argument('--dist-cache', default='/home/anqian/Desktop/my_lab/workloads/dist_cache.json')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from train_ndv import load_ndv_cache, load_dist_cache
    ndv_cache = load_ndv_cache(args.ndv_cache)
    dist_cache = load_dist_cache(args.dist_cache)
    cgroup_labels = load_cgroup_labels(args.cgroup_dir)
    print(f"Cgroup labels: {len(cgroup_labels)}")

    graphs, labels, meta = load_dataset(args.plan_dir, ndv_cache, dist_cache, cgroup_labels)
    print(f"Matched plans: {len(graphs)}")

    norm_labels, stats = normalize_labels(labels)
    key_map = {'memory_bytes': 'mem', 'network_bytes': 'net', 'disk_bytes': 'disk', 'latency_s': 'cpu'}
    for g, nl in zip(graphs, norm_labels):
        for rk, sk in key_map.items():
            setattr(g, f'y_{sk}', torch.tensor([nl[rk]], dtype=torch.float32))

    n = len(graphs)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * 0.7)]
    val_idx = indices[int(n * 0.7):int(n * 0.85)]
    test_idx = indices[int(n * 0.85):]
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64)

    model = PlanGNN(hidden_dim=128, n_layers=3, n_heads=4, dropout=0.2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf'); best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            preds = model(data)
            loss = sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1))
                       for k in key_map.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0; nb = 0
        with torch.no_grad():
            for data in val_loader:
                preds = model(data)
                val_loss += sum(F.huber_loss(preds[k].squeeze(-1), getattr(data, f'y_{k}').squeeze(-1)).item()
                                for k in key_map.values())
                nb += 1
        val_loss /= max(nb, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f'E{epoch:3d} val={val_loss:.4f} best={best_val:.4f}')

    os.makedirs('/home/anqian/Desktop/my_lab/workloads/checkpoints', exist_ok=True)
    torch.save(best_state, '/home/anqian/Desktop/my_lab/workloads/checkpoints/best_cgroup.pt')
    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64)
    rmap = {'mem': 'memory_bytes', 'disk': 'disk_bytes', 'net': 'network_bytes', 'cpu': 'latency_s'}
    all_p, all_t = {k: [] for k in key_map.values()}, {k: [] for k in key_map.values()}
    with torch.no_grad():
        for data in test_loader:
            preds = model(data)
            for k in key_map.values():
                all_p[k].append(preds[k].squeeze(-1).cpu().numpy())
                all_t[k].append(getattr(data, f'y_{k}').squeeze(-1).cpu().numpy())

    print(f"\n{'Dim':>10s} {'P50':>8s} {'P80':>8s} {'P90':>8s} {'P95':>8s} {'P99':>8s} {'R²':>8s} {'<2x':>8s} {'<5x':>8s}")
    print('-' * 85)
    for k, label in [('mem','Memory'), ('disk','DiskIO'), ('net','Network'), ('cpu','Latency')]:
        p = np.concatenate(all_p[k]).flatten()
        t = np.concatenate(all_t[k]).flatten()
        std_k = stats[rmap[k]]['std']; mn_k = stats[rmap[k]]['mean']
        p_raw = np.maximum(np.exp(p * std_k + mn_k) - 1, 0)
        t_raw = np.exp(t * std_k + mn_k) - 1
        qe = np.maximum(p_raw / np.maximum(t_raw, 1), np.maximum(t_raw, 1) / np.maximum(p_raw, 1))
        qs = np.sort(qe); nq = len(qs)
        ss_r = np.sum((t - p)**2); ss_t = np.sum((t - np.mean(t))**2)
        r2 = 1 - ss_r / max(ss_t, 1e-8)
        print(f"{label:>10s} {qs[nq//2]:>7.2f}x {qs[int(nq*0.8)]:>7.2f}x {qs[int(nq*0.9)]:>7.2f}x "
              f"{qs[int(nq*0.95)]:>7.2f}x {qs[int(nq*0.99)]:>7.2f}x {r2:>7.4f} {np.mean(qe<=2)*100:>7.0f}% {np.mean(qe<=5)*100:>7.0f}%")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Labels: cgroup physical measurements")
    print(f"Checkpoint: checkpoints/best_cgroup.pt")


if __name__ == '__main__':
    main()
