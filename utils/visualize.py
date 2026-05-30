"""
utils/visualize.py
Glass-brain overlays of disorder-relevant connectivity subgraphs using nilearn.
Saves PNG figures to FIG_DIR.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — works on Colab and SSH
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import FIG_DIR, N_ROIS, TOP_K_EDGES


def _get_aal_coords():
    """Return (116, 3) MNI coordinates for AAL ROIs."""
    from nilearn import datasets as nds
    atlas = nds.fetch_atlas_aal()
    from nilearn.image import load_img
    from nilearn.plotting import find_parcellation_cut_coords
    coords = find_parcellation_cut_coords(atlas.maps)
    return coords   # (116, 3)


def plot_glass_brain_subgraph(
    fc_matrix:   np.ndarray,       # (116, 116) raw FC
    edge_scores: np.ndarray,       # (116, 116) learned edge importance scores
    title:       str = "Subgraph",
    filename:    str = "subgraph.png",
    top_k:       int = TOP_K_EDGES,
    label:       int = 1,           # 1=ASD, 0=TC
):
    """
    Visualise top-K discriminative edges on a glass-brain overlay.
    Saves to FIG_DIR/filename.
    """
    from nilearn import plotting

    os.makedirs(FIG_DIR, exist_ok=True)
    coords = _get_aal_coords()

    # build adjacency for top-k edges
    scores_upper = np.triu(edge_scores, k=1)
    flat         = scores_upper.flatten()
    thresh_val   = np.sort(flat)[-top_k] if len(flat) >= top_k else flat.min()
    adj          = (scores_upper >= thresh_val).astype(float) * scores_upper

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    views = ["x", "y", "z"]
    cmap  = "Reds" if label == 1 else "Blues"
    label_str = "ASD" if label == 1 else "TC"

    for ax, view in zip(axes, views):
        display = plotting.plot_connectome(
            adj,
            coords,
            node_size=10,
            edge_cmap=cmap,
            edge_vmin=0,
            edge_vmax=scores_upper.max() + 1e-6,
            display_mode=view,
            axes=ax,
            annotate=False,
            colorbar=(view == "z"),
        )

    fig.suptitle(f"{title} — {label_str}  (top {top_k} edges)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out_path}")
    return out_path


def plot_fc_matrix(fc: np.ndarray, title: str = "FC matrix", filename: str = "fc.png"):
    """Quick heatmap of a raw FC matrix."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("ROI index")
    ax.set_ylabel("ROI index")
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out_path}")
    return out_path


def plot_training_curves(history: dict, filename: str = "training_curves.png"):
    """
    history keys: train_loss, val_loss, train_acc, val_acc  (lists over epochs)
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.suptitle("BrainIB training curves", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out_path}")
    return out_path


def aggregate_edge_importance(model_outputs: list, n_rois: int = N_ROIS) -> np.ndarray:
    """
    Average edge importance scores across a set of forward-pass outputs.
    model_outputs : list of dicts with 'edge_mask' (tensor) and 'edge_index' (tensor).
    Returns (n_rois, n_rois) numpy array of mean edge scores.
    """
    import torch
    score_mat = np.zeros((n_rois, n_rois))
    count_mat = np.zeros((n_rois, n_rois))

    for out in model_outputs:
        mask  = out["edge_mask"].detach().cpu().numpy()
        ei    = out["edge_index"].cpu().numpy()
        for e, (s, d) in enumerate(ei.T):
            if s < n_rois and d < n_rois:
                score_mat[s, d] += mask[e]
                count_mat[s, d] += 1

    count_mat = np.where(count_mat == 0, 1, count_mat)
    return score_mat / count_mat
