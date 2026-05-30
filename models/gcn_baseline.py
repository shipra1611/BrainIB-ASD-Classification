"""
models/gcn_baseline.py
Vanilla GCN baseline (no information bottleneck, no edge selection).
Same encoder depth and hidden dims as BrainIB for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from configs import HIDDEN_DIM, NUM_CLASSES, DROPOUT, GCN_LAYERS, N_ROIS


class VanillaGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [N_ROIS] + [HIDDEN_DIM] * GCN_LAYERS
        for i in range(GCN_LAYERS):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self.dropout = DROPOUT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        logits = self.classifier(g)
        return {"logits": logits}
