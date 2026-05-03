"""Unit tests for biodreamer.protein_dreamer.reward.ProteinRewardHead."""
from __future__ import annotations

import math

import pytest
import torch

from biodreamer.protein_dreamer.reward import ProteinRewardHead


LATENT_DIM = 64
HIDDEN_DIM = 32
BATCH = 4


@pytest.fixture
def reward_head() -> ProteinRewardHead:
    return ProteinRewardHead(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        device=torch.device("cpu"),
    )


@pytest.fixture
def z_t() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, LATENT_DIM)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_four_heads_exist(self, reward_head):
        for attr in ("stab_head", "affin_head", "act_head", "fitness_head"):
            assert hasattr(reward_head, attr)

    def test_no_express_head(self, reward_head):
        assert not hasattr(reward_head, "express_head"), (
            "express_head was renamed to fitness_head"
        )

    def test_default_weight_keys(self, reward_head):
        assert set(reward_head.weights.keys()) == {
            "stability", "affinity", "activity", "fitness"
        }

    def test_default_weights_sum_to_one(self, reward_head):
        total = sum(reward_head.weights.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# predict_multi output shapes and keys
# ---------------------------------------------------------------------------

class TestPredictMulti:
    def test_keys(self, reward_head, z_t):
        out = reward_head.predict_multi(z_t)
        assert set(out.keys()) == {"stability", "affinity", "activity", "fitness"}

    def test_output_shape(self, reward_head, z_t):
        out = reward_head.predict_multi(z_t)
        for v in out.values():
            assert v.shape == (BATCH,), f"expected ({BATCH},), got {v.shape}"

    def test_output_dtype(self, reward_head, z_t):
        out = reward_head.predict_multi(z_t)
        for v in out.values():
            assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# predict (scalar aggregation)
# ---------------------------------------------------------------------------

class TestPredict:
    def test_scalar_shape(self, reward_head, z_t):
        r = reward_head.predict(z_t)
        assert r.shape == (BATCH,)

    def test_scalar_is_weighted_sum(self, reward_head, z_t):
        multi = reward_head.predict_multi(z_t)
        expected = sum(
            reward_head.weights[k] * multi[k] for k in reward_head.weights
        )
        r = reward_head.predict(z_t)
        assert torch.allclose(r, expected.squeeze(-1), atol=1e-5)


# ---------------------------------------------------------------------------
# compute_loss — masked SmoothL1
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def _targets_all_measured(self, batch_size: int = BATCH) -> dict:
        return {
            "stability": torch.rand(batch_size),
            "affinity":  torch.rand(batch_size),
            "activity":  torch.rand(batch_size),
            "fitness":   torch.rand(batch_size),
        }

    def _targets_sparse(self, batch_size: int = BATCH) -> dict:
        nan = float("nan")
        return {
            "stability": torch.tensor([0.5, nan, nan, nan]),
            "affinity":  torch.tensor([nan, 0.3, nan, nan]),
            "activity":  torch.tensor([nan, nan, 0.7, nan]),
            "fitness":   torch.tensor([nan, nan, nan, 0.9]),
        }

    def test_loss_is_scalar(self, reward_head, z_t):
        loss = reward_head.compute_loss(z_t, self._targets_all_measured())
        assert loss.shape == ()

    def test_loss_non_negative(self, reward_head, z_t):
        loss = reward_head.compute_loss(z_t, self._targets_all_measured())
        assert loss.item() >= 0.0

    def test_loss_zero_on_perfect_prediction(self):
        # Build a head whose weights are fixed identity so we can control the output.
        head = ProteinRewardHead(latent_dim=2, hidden_dim=4, device=torch.device("cpu"))
        z = torch.zeros(1, 2)
        multi = head.predict_multi(z)
        perfect_targets = {k: v.detach().clone() for k, v in multi.items()}
        loss = head.compute_loss(z, perfect_targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_sparse_targets_only_measured_heads_contribute(self, reward_head):
        z = torch.zeros(BATCH, LATENT_DIM)
        sparse = self._targets_sparse()
        loss = reward_head.compute_loss(z, sparse)
        # If NaN masking works correctly, the loss is finite (no NaN propagation).
        assert not math.isnan(loss.item())
        assert loss.item() >= 0.0

    def test_all_nan_targets_gives_zero_loss(self, reward_head, z_t):
        all_nan = {k: torch.full((BATCH,), float("nan")) for k in
                   ("stability", "affinity", "activity", "fitness")}
        loss = reward_head.compute_loss(z_t, all_nan)
        assert loss.item() == pytest.approx(0.0, abs=1e-9)

    def test_missing_key_in_targets_is_skipped(self, reward_head, z_t):
        partial = {"stability": torch.rand(BATCH)}
        loss = reward_head.compute_loss(z_t, partial)
        assert not math.isnan(loss.item())
        assert loss.item() >= 0.0

    def test_loss_is_differentiable(self, reward_head, z_t):
        z = z_t.requires_grad_(True)
        targets = self._targets_all_measured()
        loss = reward_head.compute_loss(z, targets)
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
