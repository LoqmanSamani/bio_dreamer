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

_ENERGY_CFG = {
    "latent_dim": LATENT_DIM,
    "action_dim": ACTION_DIM,
    "transformer": {"n_layers": 2, "n_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0, "max_len": 128},
}
_DIFF_CFG = {
    "latent_dim": LATENT_DIM,
    "action_dim": ACTION_DIM,
    "ddpm": {"time_steps": 5, "schedule": "cosine",
             "denoiser": {"n_layers": 2, "n_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0}},
}






class _StubDenoiser(nn.Module):
    """Minimal denoiser with signature (xt, t, cond) → pred for DiffusionDynamics."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(xt)


def _energy_dynamics() -> EnergyBasedDynamics:
    return EnergyBasedDynamics(_ENERGY_CFG, device=torch.device("cpu"))


def _diffusion_dynamics() -> DiffusionDynamics:
    return DiffusionDynamics(
        _DIFF_CFG, denoiser=_StubDenoiser(LATENT_DIM), device=torch.device("cpu")
    )




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





class _StubPredictor(nn.Module):
    """Minimal predictor with signature (z_t, cond=...) → z_next."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(z)


class TestEnergyBasedDynamicsCustomPredictor:
    @pytest.fixture
    def dyn(self) -> EnergyBasedDynamics:
        return EnergyBasedDynamics(
            _ENERGY_CFG, predictor=_StubPredictor(LATENT_DIM), device=torch.device("cpu")
        )

    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(10)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(11)
        return torch.randn(BATCH, ACTION_DIM)

    def test_custom_predictor_is_stored(self):
        stub = _StubPredictor(LATENT_DIM)
        dyn = EnergyBasedDynamics(_ENERGY_CFG, predictor=stub, device=torch.device("cpu"))
        assert dyn.predictor is stub

    def test_predict_output_shape(self, dyn, z, a):
        assert dyn.predict(z, a).shape == (BATCH, LATENT_DIM)

    def test_predict_output_finite(self, dyn, z, a):
        assert torch.isfinite(dyn.predict(z, a)).all()

    def test_predict_output_dtype(self, dyn, z, a):
        assert dyn.predict(z, a).dtype == torch.float32

    def test_gradient_flows_through_predict(self, dyn, z, a):
        z_in = z.requires_grad_(True)
        dyn.predict(z_in, a).sum().backward()
        assert z_in.grad is not None and torch.isfinite(z_in.grad).all()




class TestDiffusionDynamicsConstruction:
    def test_auto_builds_ddpm_from_config(self):
        from biodreamer.protein_dreamer.blocks import DDPM
        dyn = DiffusionDynamics(_DIFF_CFG, device=torch.device("cpu"))
        assert isinstance(dyn.scheduler, DDPM)

    def test_custom_scheduler_is_stored(self):
        from biodreamer.protein_dreamer.blocks import DDPM
        sched = DDPM(_StubDenoiser(LATENT_DIM), {"schedule_type": "cosine", "time_steps": 5})
        dyn = DiffusionDynamics(_DIFF_CFG, scheduler=sched, device=torch.device("cpu"))
        assert dyn.scheduler is sched

    def test_denoiser_builds_ddpm_scheduler(self):
        from biodreamer.protein_dreamer.blocks import DDPM
        dyn = DiffusionDynamics(
            _DIFF_CFG, denoiser=_StubDenoiser(LATENT_DIM), device=torch.device("cpu")
        )
        assert isinstance(dyn.scheduler, DDPM)

    def test_scheduler_takes_priority_over_denoiser(self):
        from biodreamer.protein_dreamer.blocks import DDPM
        sched = DDPM(_StubDenoiser(LATENT_DIM), {"schedule_type": "linear", "time_steps": 5})
        dyn = DiffusionDynamics(
            _DIFF_CFG, scheduler=sched, denoiser=_StubDenoiser(LATENT_DIM),
            device=torch.device("cpu"),
        )
        assert dyn.scheduler is sched




class TestDiffusionDynamicsTrainingStep:
    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(20)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def z_target(self) -> torch.Tensor:
        torch.manual_seed(21)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(22)
        return torch.randn(BATCH, ACTION_DIM)

    def _ddpm_dyn(self) -> DiffusionDynamics:
        return DiffusionDynamics(
            _DIFF_CFG, denoiser=_StubDenoiser(LATENT_DIM), device=torch.device("cpu")
        )

    def _flow_dyn(self) -> DiffusionDynamics:
        from biodreamer.protein_dreamer.blocks import FlowMatching
        sched = FlowMatching(
            _StubDenoiser(LATENT_DIM), {"num_steps": 3, "solver": "euler"}
        )
        return DiffusionDynamics(_DIFF_CFG, scheduler=sched, device=torch.device("cpu"))

    def _sde_dyn(self) -> DiffusionDynamics:
        from biodreamer.protein_dreamer.blocks import SDE
        sched = SDE(
            _StubDenoiser(LATENT_DIM), {"method": "vp", "pred_type": "noise", "num_steps": 5}
        )
        return DiffusionDynamics(_DIFF_CFG, scheduler=sched, device=torch.device("cpu"))

    @pytest.mark.parametrize("dyn_factory", ["ddpm", "flow", "sde"])
    def test_training_step_returns_two_tensors(self, dyn_factory, z, z_target, a):
        dyn = {"ddpm": self._ddpm_dyn, "flow": self._flow_dyn, "sde": self._sde_dyn}[dyn_factory]()
        result = dyn.training_step(z_target, z, a)
        assert isinstance(result, tuple) and len(result) == 2

    @pytest.mark.parametrize("dyn_factory", ["ddpm", "flow", "sde"])
    def test_training_step_shapes(self, dyn_factory, z, z_target, a):
        dyn = {"ddpm": self._ddpm_dyn, "flow": self._flow_dyn, "sde": self._sde_dyn}[dyn_factory]()
        pred, target = dyn.training_step(z_target, z, a)
        assert pred.shape == (BATCH, LATENT_DIM)
        assert target.shape == (BATCH, LATENT_DIM)

    @pytest.mark.parametrize("dyn_factory", ["ddpm", "flow", "sde"])
    def test_training_step_finite(self, dyn_factory, z, z_target, a):
        dyn = {"ddpm": self._ddpm_dyn, "flow": self._flow_dyn, "sde": self._sde_dyn}[dyn_factory]()
        pred, target = dyn.training_step(z_target, z, a)
        assert torch.isfinite(pred).all()
        assert torch.isfinite(target).all()

    def test_gradient_flows_through_training_step(self, z, z_target, a):
        dyn = self._ddpm_dyn()
        z_in = z.requires_grad_(True)
        z_tgt = z_target.requires_grad_(True)
        pred, target = dyn.training_step(z_tgt, z_in, a)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        assert z_tgt.grad is not None
        assert torch.isfinite(z_tgt.grad).all()




class TestDiffusionDynamicsFlowMatching:
    @pytest.fixture
    def dyn(self) -> DiffusionDynamics:
        from biodreamer.protein_dreamer.blocks import FlowMatching
        sched = FlowMatching(
            _StubDenoiser(LATENT_DIM), {"num_steps": 3, "solver": "euler"}
        )
        return DiffusionDynamics(_DIFF_CFG, scheduler=sched, device=torch.device("cpu"))

    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(30)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(31)
        return torch.randn(BATCH, ACTION_DIM)

    def test_predict_output_shape(self, dyn, z, a):
        assert dyn.predict(z, a).shape == (BATCH, LATENT_DIM)

    def test_predict_output_finite(self, dyn, z, a):
        assert torch.isfinite(dyn.predict(z, a)).all()

    def test_noise_raises_attribute_error(self, dyn, z):
        tau = torch.zeros(BATCH, dtype=torch.long)
        with pytest.raises(AttributeError, match="does not support noise"):
            dyn.noise(z, tau)

    def test_denoise_raises_attribute_error(self, dyn, z, a):
        tau = torch.zeros(BATCH, dtype=torch.long)
        with pytest.raises(AttributeError, match="does not support denoise"):
            dyn.denoise(z, tau, z, a)

    def test_predict_distribution_shapes(self, dyn, z, a):
        dist = dyn.predict_distribution(z, a, n_samples=2)
        assert dist["mean"].shape == (BATCH, LATENT_DIM)
        assert dist["var"].shape == (BATCH, LATENT_DIM)





class TestDiffusionDynamicsSDE:
    @pytest.fixture
    def dyn(self) -> DiffusionDynamics:
        from biodreamer.protein_dreamer.blocks import SDE
        sched = SDE(
            _StubDenoiser(LATENT_DIM), {"method": "vp", "pred_type": "noise", "num_steps": 5}
        )
        return DiffusionDynamics(_DIFF_CFG, scheduler=sched, device=torch.device("cpu"))

    @pytest.fixture
    def z(self) -> torch.Tensor:
        torch.manual_seed(40)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def a(self) -> torch.Tensor:
        torch.manual_seed(41)
        return torch.randn(BATCH, ACTION_DIM)

    def test_predict_output_shape(self, dyn, z, a):
        assert dyn.predict(z, a).shape == (BATCH, LATENT_DIM)

    def test_predict_output_finite(self, dyn, z, a):
        assert torch.isfinite(dyn.predict(z, a)).all()

    def test_noise_output_shape(self, dyn, z):
        tau = torch.full((BATCH,), 0.5)
        z_noised, noise = dyn.noise(z, tau)
        assert z_noised.shape == z.shape
        assert noise.shape == z.shape

    def test_denoise_output_shape(self, dyn, z, a):
        tau = torch.full((BATCH,), 0.5)
        z_prev = dyn.denoise(z, tau, z, a)
        assert z_prev.shape == z.shape

    def test_denoise_output_finite(self, dyn, z, a):
        tau = torch.full((BATCH,), 0.5)
        z_prev = dyn.denoise(z, tau, z, a)
        assert torch.isfinite(z_prev).all()
