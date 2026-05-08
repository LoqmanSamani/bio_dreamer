from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_AA_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
_AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_LIST)}





class SIGReg(nn.Module):
    """regularises a batch of latent vectors toward N(0, I).

    penalises deviation from a standard normal in three ways:
      - per-dimension mean ≠ 0
      - per-dimension variance ≠ 1
      - random 1d projections fail mean/variance/kurtosis checks
    """
    def __init__(self, config: Any = None) -> None:
        super().__init__()
        if config is None:
            from biodreamer.protein_dreamer.config import ProteinDreamerConfig
            config = ProteinDreamerConfig().default()["training"]["sig_reg"]
        self.latent_dim  = config.get("latent_dim", 256)
        self.num_proj    = config.get("num_proj", 1024)
        self.mean_weight = config.get("mean_weight", 1.0)
        self.var_weight  = config.get("var_weight", 1.0)
        self.proj_weight = config.get("proj_weight", 1.0)
        self.eps         = config.get("eps", 1e-6)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """compute the SIGReg loss for a batch of latent vectors"""
        z = z.reshape(-1, z.shape[-1])
        device, dtype = z.device, z.dtype
        mean = z.mean(dim=0)
        var  = z.var(dim=0, unbiased=False)
        mean_loss = mean.pow(2).mean()
        var_loss  = (var - 1.0).pow(2).mean()
        # new random projections each call — stochastic coverage of the unit sphere
        w    = F.normalize(torch.randn(self.num_proj, self.latent_dim, device=device, dtype=dtype), dim=-1)
        proj = z @ w.t() # (N, num_proj)
        proj_mean = proj.mean(dim=0)
        proj_var  = proj.var(dim=0, unbiased=False)
        proj_std  = (proj_var + self.eps).sqrt()
        proj_norm = (proj - proj_mean) / proj_std

        proj_loss = (
            proj_mean.pow(2).mean()
            + (proj_var - 1.0).pow(2).mean()
            + (proj_norm.pow(4).mean(dim=0) - 3.0).pow(2).mean()  # excess kurtosis
        )

        return (
            self.mean_weight * mean_loss
            + self.var_weight  * var_loss
            + self.proj_weight * proj_loss
        )





class EMAUpdater:
    """ema update for the jepa target encoder: θ̄ ← τ·θ̄ + (1-τ)·θ.

    the target network is never included in the optimiser, this updater
    is called manually after each optimiser step.
    note: we will use only one of the EMAUpdater or the SIGReg loss, not both together.
    the SIGReg is the main regulariser for the latent space, when encoder and dynamics model are trained end-to-end.
    the EMAUpdater is an alternative regularisation strategy, where the target encoder is a slow-moving 
    average of the online encoder, and the SIGReg loss is not used.
    """
    def __init__(self, config: Any = None) -> None:
        if config is None:
            from biodreamer.protein_dreamer.config import ProteinDreamerConfig
            config = ProteinDreamerConfig().default()["training"]["ema"]
        tau = config.get("tau", 0.99)
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau

    @torch.no_grad()
    def update(self, online: nn.Module, target: nn.Module) -> None:
        """update target network parameters in-place"""
        for p_on, p_tgt in zip(online.parameters(), target.parameters()):
            p_tgt.data.mul_(self.tau).add_(p_on.data, alpha=1.0 - self.tau)





def sequence_to_aa_indices(sequence: str) -> torch.Tensor:
    """convert an amino acid string to 0-based integer indices"""
    return torch.tensor(
        [_AA_TO_IDX.get(aa.upper(), -1) for aa in sequence],
        dtype=torch.long,
    )


def aa_indices_to_sequence(indices: torch.Tensor) -> str:
    """convert integer aa indices back to a sequence string"""
    return "".join(
        _AA_LIST[i] if 0 <= i < len(_AA_LIST) else "?"
        for i in indices.tolist()
    )


def compute_hamming_distance(seq1: str, seq2: str) -> int:
    """number of positions where seq1 and seq2 differ"""
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Sequences must have equal length: {len(seq1)} vs {len(seq2)}."
        )
    return sum(a != b for a, b in zip(seq1, seq2))


def normalize_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """normalise a reward tensor to zero mean and unit variance"""
    return (rewards - rewards.mean()) / (rewards.std(correction=0) + eps)
