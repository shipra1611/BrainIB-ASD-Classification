"""
models/losses.py
Information Bottleneck loss components.

Total loss = classification_loss
           + beta  * KL_divergence      (compress Z)
           + gamma * edge_sparsity      (prune spurious edges)
"""

import torch
import torch.nn.functional as F
from configs import BETA, GAMMA


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, sigma) || N(0, I) )
    Standard closed-form for Gaussian VAE.
    Returns scalar mean over the batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def edge_sparsity_loss(edge_mask: torch.Tensor) -> torch.Tensor:
    """
    L1 penalty on edge mask scores to encourage pruning.
    edge_mask : (E,) float in [0, 1] — learned retention probability per edge.
    """
    return edge_mask.mean()


def ib_loss(
    logits:    torch.Tensor,   # (B, num_classes)
    labels:    torch.Tensor,   # (B,)
    mu:        torch.Tensor,   # (B, latent_dim)
    logvar:    torch.Tensor,   # (B, latent_dim)
    edge_mask: torch.Tensor,   # (E,) — concatenated over batch
    beta:  float = BETA,
    gamma: float = GAMMA,
):
    """
    Returns (total_loss, cls_loss, kl_loss, sparsity_loss) as scalars.
    """
    cls      = F.cross_entropy(logits, labels)
    kl       = kl_divergence(mu, logvar)
    sparsity = edge_sparsity_loss(edge_mask)
    total    = cls + beta * kl + gamma * sparsity
    return total, cls, kl, sparsity
