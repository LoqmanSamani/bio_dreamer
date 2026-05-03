from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from biodreamer.core.decoder import BaseDecoder
from biodreamer.core.dynamics import BaseDynamics
from biodreamer.core.encoder import BaseEncoder
from biodreamer.core.reward import BaseRewardHead
from biodreamer.protein_dreamer.world_model import ProteinWorldModel

LATENT_DIM = 32
ACTION_DIM = 32
BATCH      = 4
HORIZON    = 3






class _Encoder(BaseEncoder):
    def __init__(self, latent_dim: int) -> None:
        super().__init__(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def encode(self, observation: Any) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            return self.proj(observation)
        raise TypeError(f"Unexpected observation type: {type(observation)}")


class _Dynamics(BaseDynamics):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim + latent_dim, latent_dim)
        self.latent_dim = latent_dim

    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([z_t, action_emb], dim=-1))


class _RewardHead(BaseRewardHead):
    def __init__(self, latent_dim: int) -> None:
        super().__init__(latent_dim)
        self.fc = nn.Linear(latent_dim, 1)
        self.weights = {"stability": 1.0}

    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.fc(z_t).squeeze(-1)

    def compute_loss(
        self, z_t: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred = self.predict(z_t)
        y = targets.get("stability")
        if y is None:
            return torch.tensor(0.0)
        mask = ~torch.isnan(y)
        if not mask.any():
            return torch.tensor(0.0)
        return torch.nn.functional.smooth_l1_loss(pred[mask], y[mask])


class _Decoder(BaseDecoder):
    def __init__(self, latent_dim: int) -> None:
        super().__init__(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def decode(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.proj(z_t)




@pytest.fixture
def wm() -> ProteinWorldModel:
    return ProteinWorldModel(
        encoder=_Encoder(LATENT_DIM),
        dynamics=_Dynamics(ACTION_DIM),
        reward_head=_RewardHead(LATENT_DIM),
    )


@pytest.fixture
def wm_with_decoder() -> ProteinWorldModel:
    return ProteinWorldModel(
        encoder=_Encoder(LATENT_DIM),
        dynamics=_Dynamics(ACTION_DIM),
        reward_head=_RewardHead(LATENT_DIM),
        decoder=_Decoder(LATENT_DIM),
    )


@pytest.fixture
def z() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, LATENT_DIM)


@pytest.fixture
def actions() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, HORIZON, ACTION_DIM)





class TestConstruction:
    def test_is_protein_world_model(self, wm):
        assert isinstance(wm, ProteinWorldModel)

    def test_inherits_from_core_world_model(self, wm):
        from biodreamer.core.world_model import WorldModel
        assert isinstance(wm, WorldModel)

    def test_is_nn_module(self, wm):
        assert isinstance(wm, nn.Module)

    def test_decoder_defaults_to_none(self, wm):
        assert wm.decoder is None

    def test_decoder_stored_when_provided(self, wm_with_decoder):
        assert wm_with_decoder.decoder is not None

    def test_components_registered_as_submodules(self, wm):
        names = {name for name, _ in wm.named_modules()}
        assert "encoder" in names
        assert "dynamics" in names
        assert "reward_head" in names





class TestEncode:
    def test_output_shape(self, wm, z):
        z_enc = wm.encode(z)
        assert z_enc.shape == (BATCH, LATENT_DIM)

    def test_output_finite(self, wm, z):
        assert torch.isfinite(wm.encode(z)).all()

    def test_output_dtype(self, wm, z):
        assert wm.encode(z).dtype == torch.float32




class TestImagine:
    def test_states_shape(self, wm, z, actions):
        out = wm.imagine(z, actions)
        assert out["states"].shape == (BATCH, HORIZON, LATENT_DIM)

    def test_rewards_shape(self, wm, z, actions):
        out = wm.imagine(z, actions)
        assert out["rewards"].shape == (BATCH, HORIZON)

    def test_states_finite(self, wm, z, actions):
        assert torch.isfinite(wm.imagine(z, actions)["states"]).all()

    def test_rewards_finite(self, wm, z, actions):
        assert torch.isfinite(wm.imagine(z, actions)["rewards"]).all()

    def test_output_keys(self, wm, z, actions):
        assert set(wm.imagine(z, actions).keys()) == {"states", "rewards"}

    def test_horizon_1(self, wm, z):
        actions = torch.randn(BATCH, 1, ACTION_DIM)
        out = wm.imagine(z, actions)
        assert out["states"].shape == (BATCH, 1, LATENT_DIM)

    def test_different_actions_different_states(self, wm, z):
        a1 = torch.randn(BATCH, HORIZON, ACTION_DIM)
        a2 = torch.randn(BATCH, HORIZON, ACTION_DIM)
        s1 = wm.imagine(z, a1)["states"]
        s2 = wm.imagine(z, a2)["states"]
        assert not torch.allclose(s1, s2)

    def test_reward_threshold_stops_early(self, wm, z, actions):
        # Set an absurdly low threshold — rollout should stop at step 1
        out = wm.imagine(z, actions, reward_threshold=-1e9)
        assert out["states"].shape[1] == 1

    def test_reward_threshold_none_runs_full_horizon(self, wm, z, actions):
        out = wm.imagine(z, actions, reward_threshold=None)
        assert out["states"].shape[1] == HORIZON





class TestDecode:
    def test_decode_without_decoder_raises(self, wm, z):
        with pytest.raises(RuntimeError, match="no decoder"):
            wm.decode(z)

    def test_decode_with_decoder_returns_tensor(self, wm_with_decoder, z):
        out = wm_with_decoder.decode(z)
        assert isinstance(out, torch.Tensor)

    def test_decode_output_shape(self, wm_with_decoder, z):
        out = wm_with_decoder.decode(z)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_decode_output_finite(self, wm_with_decoder, z):
        assert torch.isfinite(wm_with_decoder.decode(z)).all()





class TestPredictReward:
    def test_output_shape(self, wm, z):
        r = wm.predict_reward(z)
        assert r.shape == (BATCH,)

    def test_output_finite(self, wm, z):
        assert torch.isfinite(wm.predict_reward(z)).all()

    def test_output_dtype(self, wm, z):
        assert wm.predict_reward(z).dtype == torch.float32





class TestComputeLoss:
    @pytest.fixture
    def z_target(self) -> torch.Tensor:
        torch.manual_seed(2)
        return torch.randn(BATCH, LATENT_DIM)

    @pytest.fixture
    def action_emb(self) -> torch.Tensor:
        torch.manual_seed(3)
        return torch.randn(BATCH, ACTION_DIM)

    @pytest.fixture
    def targets(self) -> Dict[str, torch.Tensor]:
        torch.manual_seed(4)
        return {"stability": torch.randn(BATCH)}

    def test_output_keys(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets)
        assert set(out.keys()) == {"loss", "loss_dyn", "loss_rew"}

    def test_total_loss_is_scalar(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets)
        assert out["loss"].ndim == 0

    def test_all_losses_finite(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets)
        for k, v in out.items():
            assert torch.isfinite(v), f"{k} is not finite"

    def test_all_losses_non_negative(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets)
        for k, v in out.items():
            assert v.item() >= 0.0, f"{k} is negative"

    def test_total_loss_combines_components(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets, beta_dyn=1.0, beta_rew=0.1)
        expected = out["loss_dyn"] + 0.1 * out["loss_rew"]
        assert torch.allclose(out["loss"], expected)

    def test_nan_targets_handled(self, wm, z, action_emb, z_target):
        all_nan = {"stability": torch.full((BATCH,), float("nan"))}
        out = wm.compute_loss(z, action_emb, z_target, all_nan)
        assert torch.isfinite(out["loss"])

    def test_gradient_flows_through_loss(self, wm, z, action_emb, z_target, targets):
        z_in = z.requires_grad_(True)
        out = wm.compute_loss(z_in, action_emb, z_target, targets)
        out["loss"].backward()
        assert z_in.grad is not None
        assert torch.isfinite(z_in.grad).all()

    def test_beta_dyn_zero_zeroes_dyn_contribution(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets, beta_dyn=0.0, beta_rew=1.0)
        assert torch.allclose(out["loss"], out["loss_rew"])

    def test_beta_rew_zero_zeroes_rew_contribution(self, wm, z, action_emb, z_target, targets):
        out = wm.compute_loss(z, action_emb, z_target, targets, beta_dyn=1.0, beta_rew=0.0)
        assert torch.allclose(out["loss"], out["loss_dyn"])
