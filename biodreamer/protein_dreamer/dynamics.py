from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from biodreamer.core.dynamics import BaseDynamics

logger = logging.getLogger(__name__)


class EnergyBasedDynamics(BaseDynamics):
    """deterministic transformer predictor (energy-based JEPA).

    ẑ_{t+1} = g_φ(z_t, a_t) where g_φ cross-attends to a projected action token.

    training objective (roadmap §4, Eq. L_B):
        L_B = β_jepa · ||ẑ_{t+1} - z̄_{t+1}||² + λ_reg · SIGReg(z_t) + β_rew · L_rew

    any nn.Module with signature predictor(z_t, cond=a_cond) → z_next can be
    substituted via the predictor argument. when predictor is None a
    DeterministicPredictor is built from the remaining hyperparameters.

    uncertainty is handled externally by EnsembleUncertainty (uncertainty.py).
    """
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        predictor: Optional[nn.Module] = None,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 1024,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.action_proj = nn.Linear(action_dim, latent_dim).to(self.device)

        if predictor is not None:
            self.predictor = predictor.to(self.device)
        else:
            from .blocks import DeterministicPredictor
            self.predictor = DeterministicPredictor(
                latent_dim=latent_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_len=max_len,
                causal=False,
            ).to(self.device)

    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """predict ẑ_{t+1} = g_φ(z_t, action_emb).

        z_t:        (B, latent_dim)
        action_emb: (B, action_dim)
        returns:    (B, latent_dim)
        """
        z_t = z_t.to(self.device)
        a_cond = self.action_proj(action_emb.to(self.device)).unsqueeze(1)
        return self.predictor(z_t, cond=a_cond)


class DiffusionDynamics(BaseDynamics):
    """conditional diffusion model in latent space (Architecture A).

    models p_θ(z_{t+1} | z_t, a_t) via iterative denoising.

    the scheduler controls the noise process. any nn.Module that implements
    the common scheduler interface can be used:
        scheduler.noise_step(x, cond)         → (pred, target)   [training]
        scheduler.sample(shape, cond)         → tensor            [inference]

    additionally, schedulers that expose DDPM-style step access support
    the noise() and denoise() methods:
        scheduler.forward_diff(x0, t, noise)  → (xt, target)
        scheduler.sample_step(xt, t, cond)    → x_prev

    built-in schedulers in blocks.py (all satisfy the interface above):
        DDPM, SDE, FlowMatchingScheduler

    when scheduler is None, a DDPM is built from the supplied denoiser nn.Module
    and the diffusion_steps / noise_schedule hyperparameters.
    when both scheduler and denoiser are None a ValueError is raised.

    training objective (roadmap §4, Eq. L_A):
        L_A = β_diff · E[||D_θ(z̄_{t+1}^τ, τ, z_t, a_t) - z̄_{t+1}||²]
              + λ_reg · SIGReg(z_t) + β_rew · L_rew

    predict_distribution() runs n_samples independent sampling chains and
    returns their empirical mean and variance for the Active Inference policy.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        scheduler: Optional[nn.Module] = None,
        denoiser: Optional[nn.Module] = None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "cosine",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.action_proj = nn.Linear(action_dim, latent_dim).to(self.device)

        if scheduler is not None:
            self.scheduler = scheduler
        elif denoiser is not None:
            from .blocks import DDPM
            self.scheduler = DDPM(
                predictor=denoiser,
                schedule_type=noise_schedule,
                time_steps=diffusion_steps,
            )
        else:
            raise ValueError(
                "DiffusionDynamics requires either 'scheduler' (a pre-built DDPM / SDE / "
                "FlowMatchingScheduler) or 'denoiser' (an nn.Module from which a default "
                "DDPM is constructed)."
            )

    def _conditioning(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """build (B, 2, latent_dim) conditioning context: [z_t token, action token]"""
        z_t = z_t.to(self.device)
        a_proj = self.action_proj(action_emb.to(self.device))
        return torch.stack([z_t, a_proj], dim=1)

    def training_step(
        self,
        z_target: torch.Tensor,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute (pred, target) for training via scheduler.noise_step().

        works with any scheduler (DDPM, SDE, FlowMatchingScheduler) since all
        implement noise_step(x, cond) → (pred, target).

        z_target:   (B, latent_dim) — clean next latent state (prediction target)
        z_t:        (B, latent_dim) — current latent state (conditioning)
        action_emb: (B, action_dim)
        returns:    (pred, target) both of shape (B, latent_dim)
        """
        cond = self._conditioning(z_t, action_emb)
        return self.scheduler.noise_step(z_target.to(self.device), cond=cond)

    def noise(self, z_clean: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """add ddpm-schedule noise at timestep tau.

        only valid when scheduler exposes forward_diff() (i.e. DDPM or SDE).
        use training_step() for scheduler-agnostic training.
        """
        if not hasattr(self.scheduler, "forward_diff"):
            raise AttributeError(
                f"{type(self.scheduler).__name__} does not support noise() — "
                "use training_step() instead."
            )
        noise = torch.randn_like(z_clean)
        z_noised, _ = self.scheduler.forward_diff(
            z_clean.to(self.device), tau.to(self.device), noise
        )
        return z_noised, noise

    def denoise(
        self,
        z_noised: torch.Tensor,
        tau: torch.Tensor,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """single reverse step z^τ → z^{τ-1}, conditioned on (z_t, action_emb).

        only valid when scheduler exposes sample_step() (i.e. DDPM or SDE).
        """
        if not hasattr(self.scheduler, "sample_step"):
            raise AttributeError(
                f"{type(self.scheduler).__name__} does not support denoise() — "
                "use predict() for full reverse sampling instead."
            )
        cond = self._conditioning(z_t, action_emb)
        return self.scheduler.sample_step(z_noised.to(self.device), tau.to(self.device), cond)

    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """full reverse pass from pure Gaussian noise to ẑ_{t+1}.

        z_t:        (B, latent_dim)
        action_emb: (B, action_dim)
        returns:    (B, latent_dim)
        """
        cond = self._conditioning(z_t, action_emb)
        B = z_t.shape[0] if z_t.dim() > 1 else 1
        return self.scheduler.sample((B, self.latent_dim), cond=cond)

    def predict_distribution(
        self,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """draw n_samples independent sampling trajectories.

        returns {"mean": (B, latent_dim), "var": (B, latent_dim)}.
        correction=0 avoids NaN when n_samples=1.
        """
        samples = torch.stack(
            [self.predict(z_t, action_emb) for _ in range(n_samples)], dim=0
        )
        return {
            "mean": samples.mean(dim=0),
            "var":  samples.var(dim=0, correction=0),
        }
