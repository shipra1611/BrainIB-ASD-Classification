# BrainIB — Interpretable GNN for Psychiatric Disorder Classification

Graph neural framework with information bottlenecking to identify minimally sufficient
functional connectivity biomarkers for ASD classification on ABIDE.

## Project layout

```
brainib/
├── data/
│   ├── download_abide.py       # Download + preprocess ABIDE FC matrices
│   └── dataset.py              # PyG Dataset class
├── models/
│   ├── brainib.py              # BrainIB encoder + subgraph selector
│   ├── gcn_baseline.py         # Vanilla GCN baseline
│   └── losses.py               # IB loss + classification loss
├── utils/
│   ├── metrics.py              # Accuracy, AUC, edge sparsity
│   └── visualize.py            # nilearn glass-brain overlays
├── train.py                    # Training script (local or Colab)
├── evaluate.py                 # Evaluation + report generation
├── configs.py                  # All hyperparameters in one place
├── notebooks/
│   └── BrainIB_Colab.ipynb     # Self-contained Colab notebook
├── outputs/
│   ├── checkpoints/            # Saved model weights
│   ├── figures/                # Glass-brain plots
│   └── results/                # Metrics JSON
└── requirements.txt
```

## Quickstart

### Option A — Google Colab (no GPU needed locally)
Open `notebooks/BrainIB_Colab.ipynb` in Colab. Everything runs cell-by-cell.
The notebook downloads data to a `/brainib_data/` folder in your Drive root — completely isolated.

### Option B — Local / VSCode
```bash
git clone <this-repo>
cd brainib
pip install -r requirements.txt
python data/download_abide.py          # ~45 min first run, cached after
python train.py                        # trains BrainIB
python evaluate.py                     # generates metrics + glass-brain plots
```

## Drive safety
Data is ALWAYS written to `/brainib_data/` at your Drive root.
Code never writes anywhere else. Delete `/brainib_data/` to clean up completely.
