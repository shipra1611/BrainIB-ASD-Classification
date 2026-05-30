"""
data/download_abide.py
Downloads ABIDE via nilearn, extracts 116-ROI (AAL) correlation matrices,
and caches them as numpy arrays.  Safe to re-run — skips completed subjects.

Usage:
    python data/download_abide.py
    python data/download_abide.py --n_subjects 50   # quick test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── allow running from project root ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import (
    DATA_DIR, CACHE_DIR, FC_DIR,
    ABIDE_PIPELINE, ABIDE_STRATEGY, ABIDE_DERIVATIVES,
    ABIDE_N_SUBJECTS, N_ROIS, CONNECTIVITY_KIND,
)


def make_dirs():
    for d in [DATA_DIR, CACHE_DIR, FC_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"[dirs] data root → {DATA_DIR}")


def fetch_abide(n_subjects=None):
    from nilearn import datasets
    print("[abide] fetching ABIDE PCP dataset …")
    abide = datasets.fetch_abide_pcp(
        data_dir=CACHE_DIR,
        pipeline=ABIDE_PIPELINE,
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=ABIDE_DERIVATIVES,
        n_subjects=n_subjects,
        verbose=1,
    )
    print(f"[abide] fetched {len(abide.func_preproc)} subjects")
    return abide


def extract_fc_matrices(abide):
    """
    Time-series → AAL parcellation → 116×116 Pearson correlation matrix.
    Saves each subject as  FC_DIR/<subject_id>.npy
    Returns a DataFrame with columns: subject_id, label (1=ASD, 0=TC), fc_path
    """
    from nilearn import datasets as nds
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    # AAL atlas
    atlas     = nds.fetch_atlas_aal()
    masker    = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        memory=CACHE_DIR,
        verbose=0,
    )
    conn_measure = ConnectivityMeasure(kind=CONNECTIVITY_KIND)

    records = []
    failed  = []

    for i, (func_path, label, sid) in enumerate(
        tqdm(
            zip(abide.func_preproc, abide.phenotypic["DX_GROUP"], abide.phenotypic["SUB_ID"]),
            total=len(abide.func_preproc),
            desc="extracting FC",
        )
    ):
        # DX_GROUP: 1=ASD, 2=TC → remap to 1/0
        label_bin = 1 if int(label) == 1 else 0
        out_path  = os.path.join(FC_DIR, f"{sid}.npy")

        if os.path.exists(out_path):
            records.append({"subject_id": sid, "label": label_bin, "fc_path": out_path})
            continue

        try:
            ts = masker.fit_transform(func_path)              # (T, 116)
            fc = conn_measure.fit_transform([ts])[0]          # (116, 116)
            np.save(out_path, fc.astype(np.float32))
            records.append({"subject_id": sid, "label": label_bin, "fc_path": out_path})
        except Exception as e:
            failed.append((sid, str(e)))

    if failed:
        print(f"[warn] {len(failed)} subjects failed extraction:")
        for sid, err in failed[:5]:
            print(f"       {sid}: {err}")

    df = pd.DataFrame(records)
    manifest_path = os.path.join(DATA_DIR, "manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"[done] {len(df)} subjects saved → manifest: {manifest_path}")
    print(f"       ASD: {df['label'].sum()}  |  TC: {(df['label']==0).sum()}")
    return df


def verify_matrices(df):
    """Quick sanity check on saved matrices."""
    sample = df.sample(min(5, len(df)))
    for _, row in sample.iterrows():
        fc = np.load(row["fc_path"])
        assert fc.shape == (N_ROIS, N_ROIS), f"unexpected shape {fc.shape}"
        assert not np.isnan(fc).any(), f"NaN in {row['subject_id']}"
    print(f"[verify] shape check passed on {len(sample)} samples — all ({N_ROIS},{N_ROIS})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=ABIDE_N_SUBJECTS,
                        help="number of subjects to fetch (None = all)")
    args = parser.parse_args()

    make_dirs()
    abide = fetch_abide(n_subjects=args.n_subjects)
    df    = extract_fc_matrices(abide)
    verify_matrices(df)
