"""
utils/metrics.py
Evaluation helpers: accuracy, AUC-ROC, edge sparsity, per-class stats.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
)


def compute_metrics(all_labels, all_probs, all_preds=None):
    """
    all_labels : list or np.array of int (0/1)
    all_probs  : list or np.array of float — P(ASD) for each sample
    all_preds  : list or np.array of int (0/1) — optional, derived from probs if None
    Returns dict of metric name → value.
    """
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    if all_preds is None:
        all_preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    report = classification_report(all_labels, all_preds,
                                   target_names=["TC", "ASD"], output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "auc_roc":  auc,
        "report":   report,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(all_labels)),
        "n_asd":     int(all_labels.sum()),
        "n_tc":      int((all_labels == 0).sum()),
    }


def edge_sparsity(hard_masks):
    """
    hard_masks : list of (E,) tensors from batch outputs
    Returns fraction of edges pruned (hard_mask < 0.5).
    """
    masks = torch.cat(hard_masks)
    pruned = (masks < 0.5).float().mean().item()
    return pruned


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}accuracy : {metrics['accuracy']:.4f}")
    print(f"{tag}AUC-ROC  : {metrics['auc_roc']:.4f}")
    r = metrics["report"]
    print(f"{tag}ASD  F1  : {r['ASD']['f1-score']:.4f}  "
          f"prec={r['ASD']['precision']:.3f}  rec={r['ASD']['recall']:.3f}")
    print(f"{tag}TC   F1  : {r['TC']['f1-score']:.4f}  "
          f"prec={r['TC']['precision']:.3f}  rec={r['TC']['recall']:.3f}")
