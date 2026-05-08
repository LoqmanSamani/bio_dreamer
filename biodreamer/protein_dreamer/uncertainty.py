"""
Epistemic Uncertainty Estimation.

Estimates the world model's uncertainty about its predictions, critical
for active inference (exploration) and active learning (deciding which
mutants to test experimentally).

Integration points:
    - core/active_inference.py: epistemic term of Expected Free Energy.
    - training/active_learning.py: candidate ranking for experimental validation.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from biodreamer.protein_dreamer.config import ProteinDreamerConfig

logger = logging.getLogger(__name__)






class UncertaintyModule(ABC):
    """Abstract interface for uncertainty estimation modules.

    All subclasses expose estimate() which returns
    (mean_prediction, epistemic_uncertainty, aleatoric_uncertainty).

    Epistemic uncertainty quantifies model uncertainty (reducible with more
    data); aleatoric quantifies irreducible observation noise.
    """

    @abstractmethod
    def estimate(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate prediction and uncertainty for a (state, action) pair.

        Args:
            z_t: Latent state, shape (B, latent_dim).
            action: Action embedding, shape (B, action_dim).

        Returns:
            mean_pred: Mean prediction, shape (B, output_dim).
            epistemic: Per-sample epistemic uncertainty, shape (B,).
            aleatoric: Per-sample aleatoric uncertainty, shape (B,).
        """

    def is_uncertain(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """Return a bool mask where epistemic uncertainty exceeds threshold.

        Args:
            z_t: Latent state, shape (B, latent_dim).
            action: Action embedding, shape (B, action_dim).
            threshold: Scalar cutoff on epistemic uncertainty.

        Returns:
            Bool tensor of shape (B,).
        """
        _, epistemic, _ = self.estimate(z_t, action)
        return epistemic > threshold

    def calibrate(self, validation_data: Any) -> None:
        """Optional post-hoc calibration on held-out data. No-op by default."""





class EnsembleUncertainty(UncertaintyModule):
    """Epistemic uncertainty from disagreement across an ensemble of models.

    Each member is any callable (z_t, action) → (B, output_dim). Variance
    across members is the epistemic uncertainty.  Typical use: train N
    separate reward heads with different seeds.

    Args:
        members: List of at least 2 callables.
    """

    def __init__(
        self,
        config: Any = None,
        members: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = (),
    ) -> None:
        if len(members) < 2:
            raise ValueError("EnsembleUncertainty requires at least 2 members")
        self.members = list(members)

    def estimate(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ensemble mean and inter-member variance.

        Epistemic uncertainty = mean variance across output dimensions.
        Aleatoric uncertainty = 0 (ensembles cannot isolate noise).
        """
        preds = torch.stack([m(z_t, action) for m in self.members], dim=0)  # (N, B, D)
        mean_pred = preds.mean(dim=0)                                         # (B, D)
        epistemic = preds.var(dim=0).mean(dim=-1)                             # (B,)
        aleatoric = torch.zeros_like(epistemic)
        return mean_pred, epistemic, aleatoric





class EvidentialUncertainty(UncertaintyModule, nn.Module):
    """Single-model uncertainty via Normal-Inverse-Gamma (NIG) evidential regression.

    The model outputs four parameters per target dimension:
        γ — predicted mean
        ν — virtual observation count (> 0)
        α — NIG shape parameter (> 1)
        β — NIG rate parameter (> 0)

    Epistemic uncertainty ∝ β / (ν · (α − 1))   [uncertainty over μ]
    Aleatoric uncertainty ∝ β / (α − 1)          [observation noise variance]

    Use nig_loss() as the training objective instead of MSE.

    Reference: Amini et al. 2020 — "Deep Evidential Regression"

    Args:
        input_dim: Dimension of concatenated [z_t, action] input.
        output_dim: Number of regression targets (e.g. 1 for scalar fitness).
        hidden_dim: Width of the two-layer MLP head.
    """

    def __init__(self, config: Any = None, net: Any = None) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = ProteinDreamerConfig().default()["uncertainty"]
        input_dim  = config.get("input_dim", 256)
        output_dim = config.get("output_dim", 1)
        hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = output_dim
        self.net = net if net is not None else nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * 4),
        )

    def _split_nig(
        self, out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.output_dim
        gamma = out[:, :D]
        nu    = F.softplus(out[:, D : 2 * D]) + 1e-6
        # α > 1 required for finite variance of the NIG distribution
        alpha = F.softplus(out[:, 2 * D : 3 * D]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3 * D :]) + 1e-6
        return gamma, nu, alpha, beta

    def forward(
        self, z_t: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return NIG parameters (γ, ν, α, β) each of shape (B, output_dim)."""
        x = torch.cat([z_t, action], dim=-1)
        return self._split_nig(self.net(x))

    def estimate(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean prediction and both uncertainty components.

        Returns:
            mean_pred: γ, shape (B, output_dim).
            epistemic: β / (ν · (α − 1)) averaged over output_dim, shape (B,).
            aleatoric: β / (α − 1) averaged over output_dim, shape (B,).
        """
        gamma, nu, alpha, beta = self.forward(z_t, action)
        epistemic = (beta / (nu * (alpha - 1))).mean(dim=-1)
        aleatoric = (beta / (alpha - 1)).mean(dim=-1)
        return gamma, epistemic, aleatoric

    def nig_loss(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        targets: torch.Tensor,
        lam: float = 1e-2,
    ) -> torch.Tensor:
        """Normal-Inverse-Gamma NLL + evidence regularisation.

        Args:
            z_t: Latent state (B, latent_dim).
            action: Action embedding (B, action_dim).
            targets: Ground-truth regression targets (B, output_dim).
            lam: Weight of the evidence regularisation term.

        Returns:
            Scalar loss.
        """
        gamma, nu, alpha, beta = self.forward(z_t, action)
        omega = 2.0 * beta * (1.0 + nu)

        nll = (
            0.5 * (math.log(math.pi) - torch.log(nu))
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (targets - gamma) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        reg = torch.abs(targets - gamma) * (2.0 * nu + alpha)
        return (nll + lam * reg).mean()





class MCDropoutUncertainty(UncertaintyModule):
    """Epistemic uncertainty via Monte Carlo Dropout at inference time.

    Runs N stochastic forward passes with dropout active, then returns
    the variance across passes as the epistemic uncertainty.  Requires the
    wrapped model to have at least one Dropout layer — otherwise uncertainty
    is always zero.

    Args:
        model: A callable (z_t, action) → (B, output_dim) with Dropout layers.
        n_passes: Number of stochastic forward passes.
    """

    def __init__(self, config: Any = None, model: Optional[nn.Module] = None) -> None:
        if config is None:
            config = ProteinDreamerConfig().default()["uncertainty"]
        n_passes = config.get("mc_dropout_passes", 20)
        if n_passes < 2:
            raise ValueError("n_passes must be at least 2")
        self.model = model
        self.n_passes = n_passes

    def estimate(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run N stochastic forward passes with dropout active.

        Returns:
            mean_pred: Mean over passes, shape (B, output_dim).
            epistemic: Variance over passes, mean over output_dim, shape (B,).
            aleatoric: Zeros (MC Dropout cannot isolate noise), shape (B,).
        """
        was_training = self.model.training
        self.model.train()  # activate dropout for all passes
        try:
            preds = torch.stack(
                [self.model(z_t, action) for _ in range(self.n_passes)],
                dim=0,
            )  # (n_passes, B, output_dim)
        finally:
            self.model.train(was_training)

        mean_pred = preds.mean(dim=0)
        epistemic  = preds.var(dim=0).mean(dim=-1)
        aleatoric  = torch.zeros(mean_pred.shape[0], device=z_t.device)
        return mean_pred, epistemic, aleatoric
