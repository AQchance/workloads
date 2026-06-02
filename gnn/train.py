"""
Training script for the PlanGNN query resource prediction model.

Usage:
    python train.py                          # default settings
    python train.py --epochs 500 --lr 1e-3   # custom hyperparams
    python train.py --split random           # random split (vs template-based)
"""

import os
import sys
import argparse
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plan_parser import parse_all_plans
from label_extractor import extract_all_labels
from model import PlanGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    plan_dir: str,
    label_dir: str,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Dict], List[str]]:
    """
    Build aligned dataset from plan files and EXPLAIN ANALYZE labels.

    Returns:
        graphs: list of PyG Data objects (on CPU for DataLoader batching)
        labels: list of label dicts (aligned with graphs)
        meta: list of "template:line_no" strings
    """
    print("Loading plans...")
    all_plans = parse_all_plans(plan_dir)

    print("Loading labels...")
    all_labels = extract_all_labels(label_dir)

    # ─── Align plans with labels by (template, line_no) ───
    graphs = []
    label_list = []
    meta_list = []

    for template in sorted(all_plans.keys()):
        plans = all_plans[template]
        # Build lookup for labels by line_no
        label_lookup = {}
        if template in all_labels:
            for entry in all_labels[template]:
                label_lookup[entry["line_no"]] = entry["labels"]

        for plan_entry in plans:
            line_no = plan_entry["line_no"]
            if line_no in label_lookup:
                graph = plan_entry["pyg_data"]
                labels = label_lookup[line_no]
                graphs.append(graph)
                label_list.append(labels)
                meta_list.append(f"{template}:{line_no}")
            else:
                print(f"  WARNING: no label for {template} line {line_no}, skipping")

    print(f"Aligned dataset: {len(graphs)} samples\n")
    return graphs, label_list, meta_list


def normalize_labels(
    label_list: List[Dict],
    stats: Dict = None,
) -> Tuple[List[Dict], Dict]:
    """
    Log-transform then z-score normalize labels per dimension.

    Log(1+x) compresses the heavy-tailed distribution of resource metrics,
    then z-score centers and scales. Applied to all four dimensions.
    """
    import math

    keys = ["cpu_time_ms", "memory_bytes", "disk_io_rows", "network_rows"]

    # Step 1: log-transform
    log_labels = []
    for lab in label_list:
        log_lab = {}
        for k in keys:
            log_lab[k] = math.log(1.0 + max(lab[k], 0.0))
        log_labels.append(log_lab)

    # Step 2: z-score on log-transformed values
    if stats is None:
        stats = {}
        for k in keys:
            vals = [l[k] for l in log_labels]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(max(var, 1e-8))
            stats[k] = {"mean": mean, "std": std}

    normalized = []
    for log_lab in log_labels:
        norm_lab = {}
        for k in keys:
            norm_lab[k] = (log_lab[k] - stats[k]["mean"]) / stats[k]["std"]
        normalized.append(norm_lab)

    return normalized, stats


def build_train_val_test_split(
    graphs: List,
    norm_labels: List[Dict],
    meta: List[str],
    split_mode: str = "template",
    val_templates: List[str] = None,
    test_templates: List[str] = None,
) -> Tuple:
    """
    Split dataset into train/val/test.

    Args:
        split_mode: "template" (split by query template) or "random" (shuffle samples)
    """
    if split_mode == "template":
        if val_templates is None:
            val_templates = ["Q13", "Q14", "Q16"]
        if test_templates is None:
            test_templates = ["Q17", "Q18", "Q19", "Q20"]

        train_idx, val_idx, test_idx = [], [], []
        for i, m in enumerate(meta):
            template = m.split(":")[0]
            if template in test_templates:
                test_idx.append(i)
            elif template in val_templates:
                val_idx.append(i)
            else:
                train_idx.append(i)

    else:  # random
        n = len(graphs)
        indices = list(range(n))
        random.shuffle(indices)
        train_idx = indices[: int(n * 0.6)]
        val_idx = indices[int(n * 0.6): int(n * 0.8)]
        test_idx = indices[int(n * 0.8):]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Attach labels to graph objects
    def attach_labels(indices):
        dataset = []
        for i in indices:
            g = graphs[i].clone()
            lab = norm_labels[i]
            g.y_mem = torch.tensor([lab["memory_bytes"]], dtype=torch.float32)
            g.y_disk = torch.tensor([lab["disk_io_rows"]], dtype=torch.float32)
            g.y_net = torch.tensor([lab["network_rows"]], dtype=torch.float32)
            g.y_cpu = torch.tensor([lab["cpu_time_ms"]], dtype=torch.float32)
            dataset.append(g)
        return dataset

    return attach_labels(train_idx), attach_labels(val_idx), attach_labels(test_idx)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Smooth L1 / Huber loss."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return (0.5 * quadratic ** 2 + delta * linear).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        preds = model(data)
        loss = (
            huber_loss(preds["mem"], data.y_mem)
            + huber_loss(preds["disk"], data.y_disk)
            + huber_loss(preds["net"], data.y_net)
            + huber_loss(preds["cpu"], data.y_cpu)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = {"mem": [], "disk": [], "net": [], "cpu": []}
    all_targets = {"mem": [], "disk": [], "net": [], "cpu": []}

    for data in loader:
        data = data.to(device)
        preds = model(data)

        loss = (
            huber_loss(preds["mem"], data.y_mem)
            + huber_loss(preds["disk"], data.y_disk)
            + huber_loss(preds["net"], data.y_net)
            + huber_loss(preds["cpu"], data.y_cpu)
        )
        total_loss += loss.item()
        n_batches += 1

        for k in all_preds:
            all_preds[k].append(preds[k].cpu().numpy())
            all_targets[k].append(getattr(data, f"y_{k}").cpu().numpy())

    # Compute R^2 per dimension
    metrics = {"loss": total_loss / max(n_batches, 1)}
    for k in all_preds:
        pred = np.concatenate(all_preds[k]).flatten()
        target = np.concatenate(all_targets[k]).flatten()
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        metrics[f"r2_{k}"] = r2
        # Also MAE
        metrics[f"mae_{k}"] = np.mean(np.abs(target - pred))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PlanGNN for query resource prediction")
    parser.add_argument("--plan-dir", type=str, default=None,
                        help="Directory with Q*_plan.txt files")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Directory with Q*_explain_analyze.txt files")
    parser.add_argument("--split", type=str, default="template",
                        choices=["template", "random"],
                        help="Train/val/test split mode")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plan_dir = args.plan_dir or os.path.join(script_dir, "..", "plans")
    label_dir = args.label_dir or os.path.join(script_dir, "..", "explain_analyze_results")
    save_dir = args.save_dir or os.path.join(script_dir, "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Build dataset ───
    graphs, labels, meta = build_dataset(plan_dir, label_dir, device)

    if len(graphs) == 0:
        print("ERROR: No aligned plan-label pairs found.")
        sys.exit(1)

    # ─── Normalize labels ───
    norm_labels, label_stats = normalize_labels(labels)
    print("Label normalization stats (after log-transform):")
    for k, v in label_stats.items():
        print(f"  {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")

    # ─── Split ───
    train_set, val_set, test_set = build_train_val_test_split(
        graphs, norm_labels, meta, split_mode=args.split,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ─── Model ───
    model = PlanGNN(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=4,
        dropout=args.dropout,
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ─── Training loop ───
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | lr={lr_now:.2e} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                  f"r2_mem={val_metrics['r2_mem']:.3f} r2_disk={val_metrics['r2_disk']:.3f} "
                  f"r2_net={val_metrics['r2_net']:.3f} r2_cpu={val_metrics['r2_cpu']:.3f}")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "label_stats": label_stats,
                "args": vars(args),
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best was {best_epoch})")
                break

    # ─── Final evaluation ───
    print("\n" + "=" * 70)
    print("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest results:")
    print(f"  Loss:    {test_metrics['loss']:.4f}")
    for k in ["mem", "disk", "net", "cpu"]:
        print(f"  R²_{k}:  {test_metrics[f'r2_{k}']:.4f}  "
              f"MAE_{k}: {test_metrics[f'mae_{k}']:.4f}")

    # Save test metrics
    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in test_metrics.items()}, f, indent=2)

    print(f"\nCheckpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
