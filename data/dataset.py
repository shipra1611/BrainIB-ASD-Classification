"""
data/dataset.py
PyTorch Geometric Dataset that loads pre-extracted FC matrices
and converts them into graphs.

Each graph:
  - nodes  : 116 ROIs
  - node features : row of the FC matrix  (116-dim, i.e. each node's connectivity profile)
  - edges  : top-(1 - threshold) strongest connections, undirected
  - label  : 1 = ASD, 0 = TC
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import (
    DATA_DIR, FC_DIR,
    N_ROIS, EDGE_THRESHOLD_PCTILE, SELF_LOOPS, SEED,
    TRAIN_RATIO, VAL_RATIO,
)


def fc_to_graph(fc: np.ndarray, threshold_pctile: int = EDGE_THRESHOLD_PCTILE) -> Data:
    """
    Convert a (116,116) FC matrix to a PyG Data object.

    Node features : each row of FC  → shape (116, 116)
    Edges         : upper-triangle pairs above the threshold percentile
    Edge weights  : absolute correlation value
    """
    n = fc.shape[0]

    # ── node features: FC row = connectivity profile of each ROI ─────────
    x = torch.tensor(fc, dtype=torch.float32)          # (116, 116)

    # ── edge construction ────────────────────────────────────────────────
    abs_fc    = np.abs(fc)
    threshold = np.percentile(abs_fc, threshold_pctile)

    rows, cols = np.where((abs_fc > threshold) & (np.tri(n, k=-1 if not SELF_LOOPS else 0) == 1))
    # make undirected: add both directions
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    edge_attr = torch.tensor(
        np.concatenate([abs_fc[rows, cols], abs_fc[rows, cols]]),
        dtype=torch.float32,
    ).unsqueeze(1)   # (E, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ABIDEDataset(Dataset):
    """
    Loads pre-extracted FC matrices from DATA_DIR/fc_matrices/
    and serves them as PyG graphs.

    Args:
        split : 'train' | 'val' | 'test' | 'all'
        transform : optional PyG transform
    """

    def __init__(self, split: str = "all", transform=None):
        super().__init__(transform=transform)

        manifest_path = os.path.join(DATA_DIR, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}\n"
                f"Run  python data/download_abide.py  first."
            )

        df = pd.read_csv(manifest_path)
        df = df.dropna(subset=["fc_path"])
        df = df[df["fc_path"].apply(os.path.exists)]

        # ── deterministic split ───────────────────────────────────────────
        rng     = np.random.RandomState(SEED)
        idx     = rng.permutation(len(df))
        n       = len(df)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        if split == "train":
            df = df.iloc[idx[:n_train]]
        elif split == "val":
            df = df.iloc[idx[n_train : n_train + n_val]]
        elif split == "test":
            df = df.iloc[idx[n_train + n_val :]]
        elif split == "all":
            pass
        else:
            raise ValueError(f"split must be train/val/test/all, got '{split}'")

        self.records = df.reset_index(drop=True)
        print(
            f"[dataset] split={split:5s}  n={len(self.records):4d}  "
            f"ASD={self.records['label'].sum():3d}  "
            f"TC={(self.records['label']==0).sum():3d}"
        )

    def len(self):
        return len(self.records)

    def get(self, idx):
        row = self.records.iloc[idx]
        fc  = np.load(row["fc_path"])
        g   = fc_to_graph(fc)
        g.y = torch.tensor([int(row["label"])], dtype=torch.long)
        g.subject_id = str(row["subject_id"])
        g.fc_raw     = torch.tensor(fc, dtype=torch.float32)   # kept for visualisation
        return g


# ── quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = ABIDEDataset(split="all")
    g  = ds[0]
    print(f"sample graph: x={g.x.shape}  edges={g.edge_index.shape}  y={g.y}")
