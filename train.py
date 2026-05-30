"""
train.py
Trains BrainIB (and optionally the GCN baseline) on the ABIDE dataset.

Usage:
    python train.py                      # trains BrainIB
    python train.py --model baseline     # trains vanilla GCN
    python train.py --beta 0.001         # override IB beta
    python train.py --n_subjects 100     # quick test with 100 subjects
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (
    CKPT_DIR, RESULTS_DIR, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    PATIENCE, BETA, GAMMA, SEED,
)
from data.dataset import ABIDEDataset
from models.brainib import BrainIB
from models.gcn_baseline import VanillaGCN
from models.losses import ib_loss
from utils.metrics import compute_metrics, edge_sparsity, print_metrics
from utils.visualize import plot_training_curves

import torch.nn.functional as F


# ── reproducibility ────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] CPU — training will be slower but works fine")
    return dev


# ── one training epoch ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, model_type="brainib", beta=BETA, gamma=GAMMA):
    model.train()
    total_loss = cls_total = kl_total = sp_total = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        logits = out["logits"]
        labels = batch.y.squeeze()

        if model_type == "brainib":
            loss, cls, kl, sp = ib_loss(
                logits, labels,
                out["mu"], out["logvar"],
                out["edge_mask"],
                beta=beta, gamma=gamma,
            )
            kl_total += kl.item()
            sp_total += sp.item()
        else:
            loss = F.cross_entropy(logits, labels)
            cls  = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        cls_total  += cls.item()
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    n = len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    return {
        "loss":    total_loss / n,
        "cls":     cls_total  / n,
        "kl":      kl_total   / n,
        "sparsity": sp_total  / n,
        "accuracy": metrics["accuracy"],
        "auc":      metrics["auc_roc"],
    }


# ── one eval epoch ─────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, device, model_type="brainib", beta=BETA, gamma=GAMMA):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    hard_masks = []

    for batch in loader:
        batch = batch.to(device)
        out    = model(batch)
        logits = out["logits"]
        labels = batch.y.squeeze()

        if model_type == "brainib":
            loss, *_ = ib_loss(
                logits, labels,
                out["mu"], out["logvar"],
                out["edge_mask"],
                beta=beta, gamma=gamma,
            )
            hard_masks.append(out["hard_mask"].cpu())
        else:
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader)
    if hard_masks:
        metrics["edge_pruned"] = edge_sparsity(hard_masks)
    return metrics


# ── main training loop ─────────────────────────────────────────────────────
def train(args):
    set_seed()
    os.makedirs(CKPT_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    # ── data ────────────────────────────────────────────────────────────
    train_ds = ABIDEDataset(split="train")
    val_ds   = ABIDEDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # ── model ───────────────────────────────────────────────────────────
    if args.model == "brainib":
        model = BrainIB().to(device)
    else:
        model = VanillaGCN().to(device)
    print(f"[model] {args.model}  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    # ── training history ─────────────────────────────────────────────────
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_auc  = 0.0
    no_improve = 0
    ckpt_path  = os.path.join(CKPT_DIR, f"{args.model}_best.pt")

    print(f"\n{'='*60}")
    print(f" Training {args.model.upper()}  —  {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, device,
                         model_type=args.model, beta=args.beta, gamma=args.gamma)
        vl = eval_epoch(model, val_loader, device,
                        model_type=args.model, beta=args.beta, gamma=args.gamma)

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(vl["accuracy"])

        scheduler.step(vl["auc_roc"])

        elapsed = time.time() - t0
        pruned  = f"  pruned={vl.get('edge_pruned', 0):.1%}" if args.model == "brainib" else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={tr['loss']:.4f} | "
            f"val_acc={vl['accuracy']:.4f}  val_auc={vl['auc_roc']:.4f}"
            f"{pruned}  [{elapsed:.1f}s]"
        )

        # ── early stopping + best checkpoint ─────────────────────────────
        if vl["auc_roc"] > best_auc:
            best_auc   = vl["auc_roc"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ new best AUC={best_auc:.4f} saved → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── save training curves ──────────────────────────────────────────────
    plot_training_curves(history, filename=f"{args.model}_training_curves.png")

    # ── save history ──────────────────────────────────────────────────────
    hist_path = os.path.join(RESULTS_DIR, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] best val AUC = {best_auc:.4f}")
    print(f"       checkpoint   → {ckpt_path}")
    return ckpt_path


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="brainib",
                        choices=["brainib", "baseline"])
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--beta",        type=float, default=BETA,
                        help="IB beta (compression strength)")
    parser.add_argument("--gamma",       type=float, default=GAMMA,
                        help="edge sparsity weight")
    args = parser.parse_args()
    train(args)
