"""Unit tests for biodreamer.protein_dreamer.dynamics."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from biodreamer.protein_dreamer.dynamics import DiffusionDynamics, EnergyBasedDynamics

LATENT_DIM = 64
ACTION_DIM = 64
BATCH = 4
HORIZON = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubDenoiser(nn.Module):
    """Minimal denoiser with signature (xt, t, cond) → pred for DiffusionDynamics."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(xt)


def _energy_dynamics() -> EnergyBasedDynamics:
    return EnergyBasedDynamics(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        n_layers=2,
        n_heads=4,
        device=torch.device("cpu"),
    )


def _diffusion_dynamics() -> DiffusionDynamics:
    return DiffusionDynamics(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        denoiser=_StubDenoiser(LATENT_DIM),
        diffusion_steps=5,  # tiny schedule for fast tests
        noise_schedule="cosine",
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# EnergyBasedDynamics
# ---------------------------------------------------------------------------

class TestEnergyBasedDynamics:
    @pytest.fixture
    def dyn(self) -> EnergyBasedDynamics:
        return _energy_dynamics()

    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(BATCH, ACTION_DIM)

    def test_predict_output_shape(self, dyn, z, a):
        z_next = dyn.predict(z, a)
        assert z_next.shape == (BATCH, LATENT_DIM)

    def test_predict_output_dtype(self, dyn, z, a):
        z_next = dyn.predict(z, a)
        assert z_next.dtype == torch.float32

    def test_predict_output_finite(self, dyn, z, a):
        z_next = dyn.predict(z, a)
        assert torch.isfinite(z_next).all()

    def test_forward_matches_predict(self, dyn, z, a):
        # forward() is the BaseDynamics alias for predict(); use eval to disable dropout
        dyn.eval()
        with torch.no_grad():
            assert torch.allclose(dyn.forward(z, a), dyn.predict(z, a))

    def test_action_conditions_prediction(self, dyn, z):
        # Different actions must produce different next states
        a1 = torch.randn(BATCH, ACTION_DIM)
        a2 = torch.randn(BATCH, ACTION_DIM)
        z1 = dyn.predict(z, a1)
        z2 = dyn.predict(z, a2)
        assert not torch.allclose(z1, z2)

    def test_rollout_output_shape(self, dyn, z):
        actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
        traj = dyn.rollout(z, actions, horizon=HORIZON)
        assert traj.shape == (BATCH, HORIZON, LATENT_DIM)

    def test_rollout_output_finite(self, dyn, z):
        actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
        traj = dyn.rollout(z, actions, horizon=HORIZON)
        assert torch.isfinite(traj).all()

    def test_rollout_horizon_matches_length(self, dyn, z):
        for h in (1, 3, 5):
            actions = torch.randn(BATCH, h, ACTION_DIM)
            traj = dyn.rollout(z, actions, horizon=h)
            assert traj.shape[1] == h

    def test_gradient_flows_through_predict(self, dyn, z, a):
        z_in = z.requires_grad_(True)
        z_next = dyn.predict(z_in, a)
        z_next.sum().backward()
        assert z_in.grad is not None
        assert torch.isfinite(z_in.grad).all()

    def test_predict_distribution_raises(self, dyn, z, a):
        # EnergyBasedDynamics delegates uncertainty to EnsembleUncertainty; no built-in distribution
        with pytest.raises(NotImplementedError):
            dyn.predict_distribution(z, a, n_samples=3)


# ---------------------------------------------------------------------------
# DiffusionDynamics
# ---------------------------------------------------------------------------

class TestDiffusionDynamics:
    @pytest.fixture
    def dyn(self) -> DiffusionDynamics:
        return _diffusion_dynamics()

    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(2)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(3)
        return torch.randn(BATCH, ACTION_DIM)

    def test_predict_output_shape(self, dyn, z, a):
        z_next = dyn.predict(z, a)
        assert z_next.shape == (BATCH, LATENT_DIM)

    def test_predict_output_finite(self, dyn, z, a):
        assert torch.isfinite(dyn.predict(z, a)).all()

    def test_forward_matches_predict(self, dyn, z, a):
        # Two separate forward() calls are stochastic — check they both return correct shape
        z1 = dyn.forward(z, a)
        z2 = dyn.predict(z, a)
        assert z1.shape == z2.shape == (BATCH, LATENT_DIM)

    def test_rollout_output_shape(self, dyn, z):
        actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
        traj = dyn.rollout(z, actions, horizon=HORIZON)
        assert traj.shape == (BATCH, HORIZON, LATENT_DIM)

    def test_predict_distribution_keys(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=3)
        assert set(dist.keys()) == {"mean", "var"}

    def test_predict_distribution_shapes(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=3)
        assert dist["mean"].shape == (BATCH, LATENT_DIM)
        assert dist["var"].shape == (BATCH, LATENT_DIM)

    def test_predict_distribution_mean_finite(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=3)
        assert torch.isfinite(dist["mean"]).all()

    def test_predict_distribution_var_non_negative(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=4)
        assert (dist["var"] >= 0).all()

    def test_predict_distribution_var_finite(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=3)
        assert torch.isfinite(dist["var"]).all()

    def test_predict_distribution_no_nan_with_one_sample(self, dyn, z, a):
        # correction=0 in var() must prevent NaN for n_samples=1
        dist = dyn.predict_distribution(z, a, n_samples=1)
        assert not torch.isnan(dist["var"]).any()

    def test_noise_output_shape(self, dyn, z):
        tau = torch.zeros(BATCH, dtype=torch.long)
        z_noised, noise = dyn.noise(z, tau)
        assert z_noised.shape == z.shape
        assert noise.shape == z.shape

    def test_noise_output_finite(self, dyn, z):
        tau = torch.randint(0, 5, (BATCH,))
        z_noised, _ = dyn.noise(z, tau)
        assert torch.isfinite(z_noised).all()

    def test_denoise_output_shape(self, dyn, z, a):
        tau = torch.ones(BATCH, dtype=torch.long)
        z_noised, _ = dyn.noise(z, tau)
        z_prev = dyn.denoise(z_noised, tau, z, a)
        assert z_prev.shape == z.shape

    def test_denoise_output_finite(self, dyn, z, a):
        tau = torch.ones(BATCH, dtype=torch.long)
        z_noised, _ = dyn.noise(z, tau)
        z_prev = dyn.denoise(z_noised, tau, z, a)
        assert torch.isfinite(z_prev).all()
