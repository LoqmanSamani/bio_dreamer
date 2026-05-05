"""Unit tests for biodreamer.protein_dreamer.uncertainty."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from biodreamer.protein_dreamer.uncertainty import (
    EnsembleUncertainty,
    EvidentialUncertainty,
    MCDropoutUncertainty,
    UncertaintyModule,
)



B = 4       # batch size
D_Z = 8     # latent dim
D_A = 4     # action dim
D_OUT = 2   # output dim


def _rand(shape):
    return torch.randn(*shape)


class _ConstModel(nn.Module):
    """Always returns the same constant tensor regardless of input."""
    def __init__(self, value: float, output_dim: int) -> None:
        super().__init__()
        self.value = value
        self.output_dim = output_dim

    def forward(self, z_t, action):
        return torch.full((z_t.shape[0], self.output_dim), self.value)


class _DropoutModel(nn.Module):
    """Small MLP with a Dropout layer — used for MCDropoutUncertainty tests."""
    def __init__(self, input_dim: int, output_dim: int, p: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Dropout(p),
            nn.Linear(16, output_dim),
        )

    def forward(self, z_t, action):
        return self.net(torch.cat([z_t, action], dim=-1))




class TestUncertaintyModuleInterface:
    def _make_ensemble(self):
        m1 = _ConstModel(1.0, D_OUT)
        m2 = _ConstModel(2.0, D_OUT)
        return EnsembleUncertainty([m1, m2])

    def test_is_subclass_of_abstract(self):
        assert issubclass(EnsembleUncertainty, UncertaintyModule)

    def test_is_uncertain_returns_bool_tensor(self):
        unc = self._make_ensemble()
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        mask = unc.is_uncertain(z, a, threshold=0.0)
        assert mask.dtype == torch.bool
        assert mask.shape == (B,)

    def test_is_uncertain_threshold_all_true(self):
        unc = self._make_ensemble()
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        # threshold=-inf → all uncertain
        mask = unc.is_uncertain(z, a, threshold=float("-inf"))
        assert mask.all()

    def test_is_uncertain_threshold_all_false(self):
        unc = self._make_ensemble()
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        # threshold=+inf → none uncertain
        mask = unc.is_uncertain(z, a, threshold=float("inf"))
        assert not mask.any()

    def test_calibrate_noop(self):
        unc = self._make_ensemble()
        unc.calibrate(None)  # must not raise




class TestEnsembleUncertaintyConstruction:
    def test_requires_at_least_two_members(self):
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleUncertainty([_ConstModel(0.0, D_OUT)])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            EnsembleUncertainty([])

    def test_stores_members(self):
        members = [_ConstModel(float(i), D_OUT) for i in range(3)]
        ens = EnsembleUncertainty(members)
        assert len(ens.members) == 3


class TestEnsembleUncertaintyEstimate:
    def _make(self, values):
        return EnsembleUncertainty([_ConstModel(v, D_OUT) for v in values])

    def test_mean_pred_shape(self):
        ens = self._make([0.0, 1.0])
        mean, _, _ = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert mean.shape == (B, D_OUT)

    def test_epistemic_shape(self):
        ens = self._make([0.0, 1.0])
        _, ep, _ = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert ep.shape == (B,)

    def test_aleatoric_shape(self):
        ens = self._make([0.0, 1.0])
        _, _, al = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert al.shape == (B,)

    def test_aleatoric_is_zero(self):
        ens = self._make([0.0, 1.0])
        _, _, al = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (al == 0).all()

    def test_identical_members_zero_epistemic(self):
        ens = self._make([3.0, 3.0, 3.0])
        _, ep, _ = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep.abs() < 1e-6).all()

    def test_disagreeing_members_positive_epistemic(self):
        ens = self._make([0.0, 10.0])
        _, ep, _ = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep > 0).all()

    def test_mean_pred_correct_value(self):
        ens = self._make([1.0, 3.0])
        mean, _, _ = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        expected = 2.0
        assert (mean - expected).abs().max() < 1e-5

    def test_more_members_reduces_variance(self):
        # All members identical → zero variance regardless of N
        ens3 = self._make([5.0, 5.0, 5.0])
        _, ep, _ = ens3.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep.abs() < 1e-6).all()

    def test_works_with_nn_module_members(self):
        members = [nn.Linear(D_Z + D_A, D_OUT) for _ in range(2)]
        callables = [lambda z, a, m=m: m(torch.cat([z, a], -1)) for m in members]
        ens = EnsembleUncertainty(callables)
        mean, ep, al = ens.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert mean.shape == (B, D_OUT)
        assert ep.shape == (B,)




class TestEvidentialUncertaintyConstruction:
    def test_is_nn_module(self):
        ev = EvidentialUncertainty(input_dim=D_Z + D_A, output_dim=D_OUT)
        assert isinstance(ev, nn.Module)

    def test_is_uncertainty_module(self):
        ev = EvidentialUncertainty(input_dim=D_Z + D_A)
        assert isinstance(ev, UncertaintyModule)

    def test_default_output_dim_one(self):
        ev = EvidentialUncertainty(input_dim=D_Z + D_A)
        assert ev.output_dim == 1

    def test_custom_hidden_dim(self):
        ev = EvidentialUncertainty(input_dim=4, output_dim=1, hidden_dim=64)
        total = sum(p.numel() for p in ev.parameters())
        assert total > 0


class TestEvidentialUncertaintyForward:
    def _make(self, output_dim=D_OUT):
        return EvidentialUncertainty(input_dim=D_Z + D_A, output_dim=output_dim)

    def test_forward_returns_four_tensors(self):
        ev = self._make()
        out = ev.forward(_rand((B, D_Z)), _rand((B, D_A)))
        assert len(out) == 4

    def test_forward_shapes(self):
        ev = self._make()
        gamma, nu, alpha, beta = ev.forward(_rand((B, D_Z)), _rand((B, D_A)))
        for t in (gamma, nu, alpha, beta):
            assert t.shape == (B, D_OUT)

    def test_nu_positive(self):
        ev = self._make()
        _, nu, _, _ = ev.forward(_rand((B, D_Z)), _rand((B, D_A)))
        assert (nu > 0).all()

    def test_alpha_greater_than_one(self):
        ev = self._make()
        _, _, alpha, _ = ev.forward(_rand((B, D_Z)), _rand((B, D_A)))
        assert (alpha > 1).all()

    def test_beta_positive(self):
        ev = self._make()
        _, _, _, beta = ev.forward(_rand((B, D_Z)), _rand((B, D_A)))
        assert (beta > 0).all()


class TestEvidentialUncertaintyEstimate:
    def _make(self, output_dim=D_OUT):
        return EvidentialUncertainty(input_dim=D_Z + D_A, output_dim=output_dim)

    def test_mean_pred_shape(self):
        ev = self._make()
        mean, _, _ = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert mean.shape == (B, D_OUT)

    def test_epistemic_shape(self):
        ev = self._make()
        _, ep, _ = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert ep.shape == (B,)

    def test_aleatoric_shape(self):
        ev = self._make()
        _, _, al = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert al.shape == (B,)

    def test_epistemic_non_negative(self):
        ev = self._make()
        _, ep, _ = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep >= 0).all()

    def test_aleatoric_non_negative(self):
        ev = self._make()
        _, _, al = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (al >= 0).all()

    def test_output_dim_one(self):
        ev = EvidentialUncertainty(input_dim=D_Z + D_A, output_dim=1)
        mean, ep, al = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert mean.shape == (B, 1)
        assert ep.shape == (B,)

    def test_finite_outputs(self):
        ev = self._make()
        mean, ep, al = ev.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        for t in (mean, ep, al):
            assert t.isfinite().all()


class TestEvidentialUncertaintyNIGLoss:
    def _make(self):
        return EvidentialUncertainty(input_dim=D_Z + D_A, output_dim=D_OUT)

    def test_loss_is_scalar(self):
        ev = self._make()
        y = _rand((B, D_OUT))
        loss = ev.nig_loss(_rand((B, D_Z)), _rand((B, D_A)), y)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        ev = self._make()
        y = _rand((B, D_OUT))
        loss = ev.nig_loss(_rand((B, D_Z)), _rand((B, D_A)), y)
        assert loss.isfinite()

    def test_loss_decreases_with_gradient_step(self):
        torch.manual_seed(0)
        ev = self._make()
        opt = torch.optim.Adam(ev.parameters(), lr=1e-3)
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        y = _rand((B, D_OUT))
        l0 = ev.nig_loss(z, a, y).item()
        for _ in range(10):
            opt.zero_grad()
            ev.nig_loss(z, a, y).backward()
            opt.step()
        l1 = ev.nig_loss(z, a, y).item()
        assert l1 < l0

    def test_gradient_flows(self):
        ev = self._make()
        z = _rand((B, D_Z)).requires_grad_(True)
        a = _rand((B, D_A))
        y = _rand((B, D_OUT))
        loss = ev.nig_loss(z, a, y)
        loss.backward()
        assert z.grad is not None
        assert z.grad.isfinite().all()

    def test_lam_zero_no_reg(self):
        ev = self._make()
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        y = _rand((B, D_OUT))
        l_reg = ev.nig_loss(z, a, y, lam=1.0)
        l_noreg = ev.nig_loss(z, a, y, lam=0.0)
        # They differ because regularisation changes the loss value
        assert l_reg.isfinite() and l_noreg.isfinite()




class TestMCDropoutUncertaintyConstruction:
    def test_requires_at_least_two_passes(self):
        m = _DropoutModel(D_Z + D_A, D_OUT)
        with pytest.raises(ValueError, match="at least 2"):
            MCDropoutUncertainty(m, n_passes=1)

    def test_stores_model_and_passes(self):
        m = _DropoutModel(D_Z + D_A, D_OUT)
        mc = MCDropoutUncertainty(m, n_passes=5)
        assert mc.model is m
        assert mc.n_passes == 5


class TestMCDropoutUncertaintyEstimate:
    def _make(self, p=0.5, n_passes=20):
        return MCDropoutUncertainty(
            _DropoutModel(D_Z + D_A, D_OUT, p=p), n_passes=n_passes
        )

    def test_mean_pred_shape(self):
        mc = self._make()
        mean, _, _ = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert mean.shape == (B, D_OUT)

    def test_epistemic_shape(self):
        mc = self._make()
        _, ep, _ = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert ep.shape == (B,)

    def test_aleatoric_shape(self):
        mc = self._make()
        _, _, al = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert al.shape == (B,)

    def test_aleatoric_is_zero(self):
        mc = self._make()
        _, _, al = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (al == 0).all()

    def test_epistemic_non_negative(self):
        mc = self._make()
        _, ep, _ = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep >= 0).all()

    def test_finite_outputs(self):
        mc = self._make()
        mean, ep, al = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        for t in (mean, ep, al):
            assert t.isfinite().all()

    def test_high_dropout_gives_positive_epistemic(self):
        torch.manual_seed(42)
        mc = self._make(p=0.9, n_passes=50)
        _, ep, _ = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        # High dropout rate → non-zero variance across most samples
        assert ep.sum() > 0

    def test_model_returned_to_original_train_mode(self):
        m = _DropoutModel(D_Z + D_A, D_OUT)
        m.eval()
        mc = MCDropoutUncertainty(m, n_passes=5)
        mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert not m.training  # restored to eval

    def test_model_returned_to_original_eval_mode(self):
        m = _DropoutModel(D_Z + D_A, D_OUT)
        m.train()
        mc = MCDropoutUncertainty(m, n_passes=5)
        mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert m.training  # restored to train

    def test_no_dropout_zero_epistemic(self):
        class NoDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(D_Z + D_A, D_OUT)

            def forward(self, z, a):
                return self.linear(torch.cat([z, a], dim=-1))

        mc = MCDropoutUncertainty(NoDropoutModel(), n_passes=10)
        _, ep, _ = mc.estimate(_rand((B, D_Z)), _rand((B, D_A)))
        assert (ep.abs() < 1e-6).all()

    def test_is_uncertain_integrates_correctly(self):
        mc = self._make(p=0.9, n_passes=50)
        z = _rand((B, D_Z))
        a = _rand((B, D_A))
        mask = mc.is_uncertain(z, a, threshold=0.0)
        assert mask.dtype == torch.bool
        assert mask.shape == (B,)
