"""
configs.py — all hyperparameters in one place.
Edit here; everything else imports from here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
# On Colab, DRIVE_ROOT is set to '/content/drive/MyDrive'
# Locally, it defaults to a local ./data_cache folder
# Data is ALWAYS isolated inside DRIVE_ROOT/brainib_data/
DRIVE_ROOT   = os.environ.get("DRIVE_ROOT", "./data_cache")
DATA_DIR     = os.path.join(DRIVE_ROOT, "brainib_data")        # ← only folder we ever touch
CACHE_DIR    = os.path.join(DATA_DIR, "nilearn_cache")
FC_DIR       = os.path.join(DATA_DIR, "fc_matrices")
CKPT_DIR     = os.path.join(DATA_DIR, "checkpoints")
FIG_DIR      = os.path.join(DATA_DIR, "figures")
RESULTS_DIR  = os.path.join(DATA_DIR, "results")

# ── ABIDE fetch settings ───────────────────────────────────────────────────
ABIDE_PIPELINE       = "cpac"
ABIDE_STRATEGY       = "filt_global"          # band-pass filtered, global signal regressed
ABIDE_DERIVATIVES    = ["func_preproc"]
ABIDE_N_SUBJECTS     = None                   # None = all 871; set e.g. 100 for quick test
ATLAS                = "schaefer100"                  # AAL atlas → 116 ROIs
N_ROIS               = 100
CONNECTIVITY_KIND    = "correlation"          # pearson correlation as FC measure

# ── Graph construction ─────────────────────────────────────────────────────
EDGE_THRESHOLD_PCTILE = 70                    # keep top-30% strongest connections
SELF_LOOPS            = False

# ── Model ──────────────────────────────────────────────────────────────────
HIDDEN_DIM   = 64
LATENT_DIM   = 32
NUM_CLASSES  = 2                              # ASD vs. typical control
DROPOUT      = 0.3
GCN_LAYERS   = 3

# ── IB objective ──────────────────────────────────────────────────────────
BETA         = 0.01   # IB trade-off  (try 0.001, 0.01, 0.1)
GAMMA        = 0.5    # edge sparsity regularisation weight

# ── Training ──────────────────────────────────────────────────────────────
EPOCHS       = 100
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20     # early stopping patience
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

# ── Visualization ─────────────────────────────────────────────────────────
TOP_K_EDGES  = 30     # show top-K discriminative connections in glass-brain
