"""
evaluate.py
Loads a trained checkpoint and evaluates on the held-out test set.
Generates:
  - classification metrics (printed + saved to JSON)
  - glass-brain connectivity overlays for ASD and TC subjects
  - FC matrix heatmaps for representative subjects

Usage:
    python evaluate.py                        # evaluates BrainIB
    python evaluate.py --model baseline       # evaluates GCN baseline
    python evaluate.py --model brainib --model baseline   # compare both
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import CKPT_DIR, RESULTS_DIR, N_ROIS, BATCH_SIZE
from data.dataset import ABIDEDataset
from models.brainib import BrainIB
from models.gcn_baseline import VanillaGCN
from utils.metrics import compute_metrics, edge_sparsity, print_metrics
from utils.visualize import (
    plot_glass_brain_subgraph,
    plot_fc_matrix,
    aggregate_edge_importance,
)


def load_model(model_type: str, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{model_type}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run  python train.py --model {model_type}  first."
        )
    model = BrainIB() if model_type == "brainib" else VanillaGCN()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[eval] loaded {model_type} from epoch {ckpt['epoch']}  (best AUC={ckpt['best_auc']:.4f})")
    return model


@torch.no_grad()
def run_eval(model, loader, device, model_type="brainib"):
    all_labels, all_probs, all_preds = [], [], []
    outputs_asd, outputs_tc = [], []
    hard_masks = []

    for batch in loader:
        batch  = batch.to(device)
        out    = model(batch)
        logits = out["logits"]
        labels = batch.y.squeeze().cpu().numpy()
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = (probs > 0.5).astype(int)

        all_labels.extend(labels)
        all_probs.extend(probs)
        all_preds.extend(preds)

        if model_type == "brainib":
            hard_masks.append(out["hard_mask"].cpu())
            # collect per-subject outputs for visualisation
            for i, lbl in enumerate(labels):
                rec = {
                    "edge_mask":   out["edge_mask"].cpu(),
                    "edge_index":  batch.edge_index.cpu(),
                    "fc_raw":      batch.fc_raw[i].cpu().numpy() if hasattr(batch, "fc_raw") else None,
                }
                if lbl == 1:
                    outputs_asd.append(rec)
                else:
                    outputs_tc.append(rec)

    metrics = compute_metrics(all_labels, all_probs, all_preds)
    if hard_masks:
        metrics["edge_pruned"] = edge_sparsity(hard_masks)

    return metrics, outputs_asd, outputs_tc


def evaluate(model_types: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds     = ABIDEDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=False)

    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f" Evaluating {model_type.upper()}")
        print(f"{'='*60}")

        model = load_model(model_type, device)
        metrics, outputs_asd, outputs_tc = run_eval(model, test_loader, device, model_type)

        print_metrics(metrics, prefix=model_type)
        if "edge_pruned" in metrics:
            print(f"[{model_type}] edges pruned : {metrics['edge_pruned']:.1%}")

        all_results[model_type] = metrics

        # ── glass-brain plots (BrainIB only) ─────────────────────────────
        if model_type == "brainib":
            print("[viz] aggregating edge importance scores …")

            if outputs_asd:
                score_asd = aggregate_edge_importance(outputs_asd, n_rois=N_ROIS)
                plot_glass_brain_subgraph(
                    fc_matrix=outputs_asd[0]["fc_raw"] if outputs_asd[0]["fc_raw"] is not None else score_asd,
                    edge_scores=score_asd,
                    title="ASD discriminative subgraph",
                    filename="glass_brain_ASD.png",
                    label=1,
                )

            if outputs_tc:
                score_tc = aggregate_edge_importance(outputs_tc, n_rois=N_ROIS)
                plot_glass_brain_subgraph(
                    fc_matrix=outputs_tc[0]["fc_raw"] if outputs_tc[0]["fc_raw"] is not None else score_tc,
                    edge_scores=score_tc,
                    title="TC discriminative subgraph",
                    filename="glass_brain_TC.png",
                    label=0,
                )

            # sample FC matrix heatmap
            if outputs_asd and outputs_asd[0]["fc_raw"] is not None:
                plot_fc_matrix(outputs_asd[0]["fc_raw"], title="Sample ASD FC matrix",
                               filename="fc_sample_ASD.png")

    # ── save all metrics ─────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "test_metrics.json")

    # convert numpy types to python native for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, (np.ndarray,)):    return obj.tolist()
        if isinstance(obj, dict):             return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):             return [_convert(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\n[done] metrics saved → {out_path}")

    # ── comparison table ──────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'Model':<12} {'Accuracy':>10} {'AUC-ROC':>10} {'Pruned':>10}")
        print("-" * 46)
        for name, m in all_results.items():
            pruned = f"{m.get('edge_pruned', 0):.1%}"
            print(f"{name:<12} {m['accuracy']:>10.4f} {m['auc_roc']:>10.4f} {pruned:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+",
                        default=["brainib"],
                        choices=["brainib", "baseline"],
                        help="model(s) to evaluate")
    args = parser.parse_args()
    evaluate(args.model)
