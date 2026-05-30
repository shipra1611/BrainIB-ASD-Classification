"""
models/brainib.py
Brain Information Bottleneck (BrainIB) model.

Architecture:
  1. GCNEncoder    – 3-layer GCN → graph-level mu and logvar
  2. Reparameterise – z = mu + eps * sigma
  3. EdgeSelector  – learns a scalar mask per edge → prunes spurious connections
  4. Classifier    – linear head on pooled subgraph embedding → ASD vs TC

Reference:
  "BrainIB: Interpretable Brain Network-based Psychiatric Diagnosis with
   Graph Information Bottleneck" (NeurIPS workshop 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from configs import HIDDEN_DIM, LATENT_DIM, NUM_CLASSES, DROPOUT, GCN_LAYERS, N_ROIS


# ─────────────────────────────────────────────────────────────────────────────
class GCNEncoder(nn.Module):
    """
    Multi-layer GCN that produces graph-level mu and log-variance
    for the information bottleneck latent variable Z.
    """

    def __init__(self, in_channels: int = N_ROIS, hidden: int = HIDDEN_DIM,
                 latent: int = LATENT_DIM, n_layers: int = GCN_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_channels] + [hidden] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = dropout
        # produce mu and log-variance from pooled representation
        self.mu_head    = nn.Linear(hidden * 2, latent)   # *2 for mean+max pool
        self.logvar_head = nn.Linear(hidden * 2, latent)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level readout: concatenate mean and max pooling
        g_mean = global_mean_pool(x, batch)   # (B, hidden)
        g_max  = global_max_pool(x, batch)    # (B, hidden)
        g      = torch.cat([g_mean, g_max], dim=-1)  # (B, hidden*2)

        mu     = self.mu_head(g)              # (B, latent)
        logvar = self.logvar_head(g)          # (B, latent)
        return mu, logvar, x                  # also return node embeddings


# ─────────────────────────────────────────────────────────────────────────────
class EdgeSelector(nn.Module):
    """
    Learns a scalar mask in [0,1] for each edge.
    Edges with low scores are treated as spurious and pruned.

    Input per edge: concatenation of source and target node embeddings.
    Output: soft mask in (0,1) via sigmoid, hard binary mask via straight-through.
    """

    def __init__(self, node_dim: int = HIDDEN_DIM, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor):
        """
        node_emb   : (N_total, node_dim)
        edge_index : (2, E)
        returns
          soft_mask : (E,) float in (0,1) — used in IB loss
          hard_mask : (E,) {0,1} — used for subgraph construction (straight-through)
        """
        src_emb = node_emb[edge_index[0]]   # (E, node_dim)
        dst_emb = node_emb[edge_index[1]]   # (E, node_dim)
        cat     = torch.cat([src_emb, dst_emb], dim=-1)  # (E, node_dim*2)
        logit   = self.mlp(cat).squeeze(-1)               # (E,)
        soft    = torch.sigmoid(logit)                    # (E,) soft mask

        # straight-through estimator: hard in forward, soft gradient in backward
        hard    = (soft > 0.5).float()
        hard    = hard - soft.detach() + soft             # straight-through

        return soft, hard


# ─────────────────────────────────────────────────────────────────────────────
class BrainIB(nn.Module):
    """
    Full BrainIB model.

    Forward returns a dict with keys:
      logits      (B, num_classes)
      mu          (B, latent_dim)
      logvar      (B, latent_dim)
      edge_mask   (E,)  — soft masks for IB loss
      hard_mask   (E,)  — hard binary masks (0=pruned, 1=kept)
    """

    def __init__(self):
        super().__init__()
        self.encoder   = GCNEncoder()
        self.selector  = EdgeSelector(node_dim=HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # use mean at inference

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        batch      = data.batch

        # ── 1. encode ─────────────────────────────────────────────────────
        mu, logvar, node_emb = self.encoder(x, edge_index, batch)
        z                    = self.reparameterise(mu, logvar)

        # ── 2. edge selection ─────────────────────────────────────────────
        soft_mask, hard_mask = self.selector(node_emb, edge_index)

        # ── 3. masked pooling for classification ──────────────────────────
        # weight each node's contribution by the mean of its adjacent edge masks
        node_weight = self._scatter_edge_to_node(soft_mask, edge_index, x.size(0))
        weighted_emb = node_emb * node_weight.unsqueeze(-1)
        g_masked = global_mean_pool(weighted_emb, batch)   # (B, hidden)

        # blend IB latent with masked graph embedding for final classification
        combined = z + global_mean_pool(node_emb, batch)   # skip-connection feel
        logits   = self.classifier(combined)

        return {
            "logits":    logits,
            "mu":        mu,
            "logvar":    logvar,
            "edge_mask": soft_mask,
            "hard_mask": hard_mask,
            "node_emb":  node_emb,
        }

    @staticmethod
    def _scatter_edge_to_node(edge_mask, edge_index, n_nodes):
        """Average edge mask score per node (for weighted pooling)."""
        src = edge_index[0]
        node_score = torch.zeros(n_nodes, device=edge_mask.device)
        node_count = torch.zeros(n_nodes, device=edge_mask.device)
        node_score.scatter_add_(0, src, edge_mask)
        node_count.scatter_add_(0, src, torch.ones_like(edge_mask))
        node_count = node_count.clamp(min=1)
        return node_score / node_count


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    def make_fake_graph(n=116, label=0):
        fc = torch.randn(n, n)
        ei = torch.randint(0, n, (2, 300))
        return Data(x=fc, edge_index=ei, y=torch.tensor([label]))

    graphs = [make_fake_graph(label=i % 2) for i in range(4)]
    batch  = Batch.from_data_list(graphs)

    model = BrainIB()
    out   = model(batch)
    print("logits :", out["logits"].shape)
    print("mu     :", out["mu"].shape)
    print("mask   :", out["edge_mask"].shape)
    pruned = (out["hard_mask"] < 0.5).float().mean().item()
    print(f"pruned edges: {pruned*100:.1f}%")
