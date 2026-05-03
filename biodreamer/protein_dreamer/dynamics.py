from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from biodreamer.core.dynamics import BaseDynamics

logger = logging.getLogger(__name__)


class EnergyBasedDynamics(BaseDynamics):
    """Architecture B: deterministic Transformer predictor (Energy-Based JEPA).

    ẑ_{t+1} = g_φ(z_t, a_t) where g_φ is a DeterministicPredictor whose
    self-attention operates on z_t and cross-attends to a projected action token.

    Training objective (roadmap §4, Eq. L_B):
        L_B = β_jepa · ||ẑ_{t+1} - sg(z̄_{t+1})||² + λ_reg · SIGReg(z_t) + β_rew · L_rew

    Uncertainty for this architecture is handled externally by EnsembleUncertainty
    (uncertainty.py), not by repeated calls to predict().
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
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
        # Project action_emb → latent_dim so the action token has the same d_model as z_t
        self.action_proj = nn.Linear(action_dim, latent_dim).to(self.device)
        from .nets import DeterministicPredictor
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
        """Predict ẑ_{t+1} = g_φ(z_t, action_emb).

        z_t:        (B, latent_dim)
        action_emb: (B, action_dim)
        Returns:    (B, latent_dim)
        """
        z_t = z_t.to(self.device)
        # Action token for cross-attention: (B, latent_dim) → (B, 1, latent_dim)
        a_cond = self.action_proj(action_emb.to(self.device)).unsqueeze(1)
        return self.predictor(z_t, cond=a_cond)


class DiffusionDynamics(BaseDynamics):
    """Architecture A: conditional DDPM in latent space (Latent Diffusion JEPA).

    Models p_θ(z_{t+1} | z_t, a_t) via iterative denoising. The denoiser must
    accept (xt: Tensor, t: Tensor, cond: Tensor) → pred: Tensor, matching the
    interface expected by DDPM in nets.py.

    Training objective (roadmap §4, Eq. L_A):
        L_A = β_diff · E[||D_θ(z̄_{t+1}^τ, τ, z_t, a_t) - z̄_{t+1}||²]
              + λ_reg · SIGReg(z_t) + β_rew · L_rew

    Uncertainty is built-in: predict_distribution() runs n_samples independent
    reverse diffusion chains and returns their empirical mean and variance, giving
    the epistemic signal for the Active Inference policy (roadmap §6).
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        denoiser: nn.Module,
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
        from .nets import DDPM
        self.ddpm = DDPM(
            predictor=denoiser,
            schedule_type=noise_schedule,
            time_steps=diffusion_steps,
            device=self.device,
        )

    def _conditioning(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Build (B, 2, latent_dim) conditioning context: [z_t token, action token]."""
        z_t = z_t.to(self.device)
        a_proj = self.action_proj(action_emb.to(self.device))  # (B, latent_dim)
        return torch.stack([z_t, a_proj], dim=1)               # (B, 2, latent_dim)

    def noise(self, z_clean: torch.Tensor, tau: torch.Tensor):
        """Add cosine-schedule diffusion noise at step tau.

        Returns (z_noised, noise) for computing the training loss L_A.
        """
        noise = torch.randn_like(z_clean)
        z_noised, _ = self.ddpm.forward_diff(
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
        """Single DDPM reverse step z^τ → z^{τ-1}, conditioned on (z_t, action_emb)."""
        cond = self._conditioning(z_t, action_emb)
        return self.ddpm.sample_step(z_noised.to(self.device), tau.to(self.device), cond)

    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Full reverse diffusion from pure Gaussian noise to ẑ_{t+1}.

        z_t:        (B, latent_dim)
        action_emb: (B, action_dim)
        Returns:    (B, latent_dim)
        """
        cond = self._conditioning(z_t, action_emb)
        B = z_t.shape[0] if z_t.dim() > 1 else 1
        return self.ddpm.sample((B, self.latent_dim), cond=cond)

    def predict_distribution(
        self,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """Draw n_samples independent reverse diffusion trajectories.

        Returns {"mean": (B, latent_dim), "var": (B, latent_dim)}.
        variance uses correction=0 (population variance) to avoid NaN when n_samples=1.
        """
        samples = torch.stack(
            [self.predict(z_t, action_emb) for _ in range(n_samples)], dim=0
        )  # (n_samples, B, latent_dim)
        return {
            "mean": samples.mean(dim=0),
            "var":  samples.var(dim=0, correction=0),
        }
