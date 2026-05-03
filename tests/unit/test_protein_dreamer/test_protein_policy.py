from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from biodreamer.core.active_inference import ActiveInferencePolicy
from biodreamer.core.policy import BasePolicy
from biodreamer.protein_dreamer.encoder import ActionEncoder
from biodreamer.protein_dreamer.policy import ProteinActiveInferencePolicy

LATENT_DIM = 32
EMBED_DIM  = 16
SEQ_LEN    = 5
N_AA       = 20
N_CANDS    = SEQ_LEN * N_AA
BATCH      = 3
HORIZON    = 4






class _DynamicsStub:
    """plain object (not nn.Module), world model is stored as a plain attribute"""

    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        # mix in the action so different (z_t, action) -> different z_next
        return z_t + action_emb[:, : z_t.shape[-1]]

    def predict_distribution(self, *args, **kwargs):
        raise NotImplementedError


class _RewardHeadStub:
    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        return z_t.mean(dim=-1)


class _WorldModelStub:
    def __init__(self) -> None:
        self.dynamics    = _DynamicsStub()
        self.reward_head = _RewardHeadStub()





@pytest.fixture
def ae() -> ActionEncoder:
    return ActionEncoder(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        device=torch.device("cpu"),
    )


@pytest.fixture
def wm() -> _WorldModelStub:
    return _WorldModelStub()


@pytest.fixture
def policy(ae, wm) -> ProteinActiveInferencePolicy:
    return ProteinActiveInferencePolicy(
        latent_dim=LATENT_DIM,
        action_dim=LATENT_DIM,
        world_model=wm,
        action_encoder=ae,
        seq_len=SEQ_LEN,
        n_aa=N_AA,
        eta=1.0,
        n_samples=2,
        device=torch.device("cpu"),
    )


@pytest.fixture
def z() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, LATENT_DIM)





class TestConstruction:
    def test_is_protein_active_inference_policy(self, policy):
        assert isinstance(policy, ProteinActiveInferencePolicy)

    def test_inherits_active_inference_policy(self, policy):
        assert isinstance(policy, ActiveInferencePolicy)

    def test_inherits_base_policy(self, policy):
        assert isinstance(policy, BasePolicy)

    def test_is_nn_module(self, policy):
        assert isinstance(policy, nn.Module)

    def test_action_encoder_registered_as_submodule(self, policy):
        names = {name for name, _ in policy.named_modules()}
        assert "action_encoder" in names

    def test_seq_len_stored(self, policy):
        assert policy.seq_len == SEQ_LEN

    def test_n_aa_stored(self, policy):
        assert policy.n_aa == N_AA

    def test_positions_buffer_shape(self, policy):
        assert policy._positions.shape == (N_CANDS,)

    def test_aa_new_buffer_shape(self, policy):
        assert policy._aa_new.shape == (N_CANDS,)

    def test_wt_aa_buffer_shape(self, policy):
        assert policy._wt_aa.shape == (SEQ_LEN,)

    def test_wt_aa_initialised_to_zeros(self, policy):
        assert (policy._wt_aa == 0).all()

    def test_positions_buffer_values(self, policy):
        # position 0 repeated N_AA times, then position 1, etc.
        expected = torch.arange(SEQ_LEN, dtype=torch.long).repeat_interleave(N_AA)
        assert torch.equal(policy._positions, expected)

    def test_aa_new_buffer_values(self, policy):
        expected = torch.arange(N_AA, dtype=torch.long).repeat(SEQ_LEN)
        assert torch.equal(policy._aa_new, expected)




class TestSetSequence:
    def test_updates_wt_aa(self, policy):
        new_seq = torch.tensor([1, 5, 3, 7, 11], dtype=torch.long)
        policy.set_sequence(new_seq)
        assert torch.equal(policy._wt_aa, new_seq)

    def test_shape_preserved(self, policy):
        policy.set_sequence(torch.arange(SEQ_LEN, dtype=torch.long))
        assert policy._wt_aa.shape == (SEQ_LEN,)

    def test_device_transferred(self, policy):
        new_seq = torch.arange(SEQ_LEN, dtype=torch.long)
        policy.set_sequence(new_seq)
        assert policy._wt_aa.device.type == "cpu"




class TestGenerateCandidateActions:
    def test_output_shape(self, policy, z):
        cands = policy._generate_candidate_actions(z)
        assert cands.shape == (BATCH, N_CANDS, LATENT_DIM)

    def test_output_finite(self, policy, z):
        assert torch.isfinite(policy._generate_candidate_actions(z)).all()

    def test_output_dtype(self, policy, z):
        assert policy._generate_candidate_actions(z).dtype == torch.float32

    def test_batch_size_1(self, policy):
        z = torch.randn(1, LATENT_DIM)
        cands = policy._generate_candidate_actions(z)
        assert cands.shape == (1, N_CANDS, LATENT_DIM)

    def test_different_wt_aa_different_embeddings(self, policy, z):
        cands_default = policy._generate_candidate_actions(z).clone()
        policy.set_sequence(torch.ones(SEQ_LEN, dtype=torch.long) * 5)
        cands_new = policy._generate_candidate_actions(z)
        assert not torch.allclose(cands_default, cands_new)

    def test_same_wt_aa_same_embeddings(self, policy, z):
        cands1 = policy._generate_candidate_actions(z)
        cands2 = policy._generate_candidate_actions(z)
        assert torch.allclose(cands1, cands2)

    def test_all_n_aa_positions_covered(self, policy, z):
        # Each of the SEQ_LEN positions should generate N_AA distinct aa_new embeddings
        cands = policy._generate_candidate_actions(z)
        # Position 0 candidates are the first N_AA rows
        pos0 = cands[0, :N_AA, :]
        # They should not all be identical (different aa_new indices → different embeddings)
        assert not torch.allclose(pos0[0], pos0[1])





class TestEvaluateActions:
    @pytest.fixture
    def candidates(self, policy, z) -> torch.Tensor:
        return policy._generate_candidate_actions(z)

    def test_output_keys(self, policy, z, candidates):
        out = policy.evaluate_actions(z, candidates)
        assert set(out.keys()) == {"efe", "rewards", "uncertainties"}

    def test_efe_shape(self, policy, z, candidates):
        out = policy.evaluate_actions(z, candidates)
        assert out["efe"].shape == (BATCH, N_CANDS)

    def test_rewards_shape(self, policy, z, candidates):
        out = policy.evaluate_actions(z, candidates)
        assert out["rewards"].shape == (BATCH, N_CANDS)

    def test_uncertainties_shape(self, policy, z, candidates):
        out = policy.evaluate_actions(z, candidates)
        assert out["uncertainties"].shape == (BATCH, N_CANDS)

    def test_efe_finite(self, policy, z, candidates):
        assert torch.isfinite(policy.evaluate_actions(z, candidates)["efe"]).all()

    def test_uncertainties_zero_for_deterministic_dynamics(self, policy, z, candidates):
        # _DynamicsStub raises NotImplementedError → fallback to zeros
        unc = policy.evaluate_actions(z, candidates)["uncertainties"]
        assert (unc == 0).all()





class TestSelectAction:
    def test_output_shape(self, policy, z):
        a = policy.select_action(z)
        assert a.shape == (BATCH, LATENT_DIM)

    def test_output_finite(self, policy, z):
        assert torch.isfinite(policy.select_action(z)).all()

    def test_output_dtype(self, policy, z):
        assert policy.select_action(z).dtype == torch.float32

    def test_forward_delegates_to_select_action(self, policy, z):
        # BasePolicy.forward() calls select_action()
        policy.eval()
        with torch.no_grad():
            assert torch.allclose(policy(z), policy.select_action(z))

    def test_different_states_may_differ(self, policy):
        z1 = torch.randn(BATCH, LATENT_DIM)
        z2 = torch.randn(BATCH, LATENT_DIM)
        a1 = policy.select_action(z1)
        a2 = policy.select_action(z2)
        assert not torch.allclose(a1, a2)




class TestSelectActionWithExploration:
    def test_output_shape(self, policy, z):
        a = policy.select_action_with_exploration(z)
        assert a.shape == (BATCH, LATENT_DIM)

    def test_output_finite(self, policy, z):
        assert torch.isfinite(policy.select_action_with_exploration(z)).all()

    def test_output_dtype(self, policy, z):
        assert policy.select_action_with_exploration(z).dtype == torch.float32

    def test_returns_a_valid_candidate(self, policy, z):
        # The returned action must be one of the generated candidates
        cands = policy._generate_candidate_actions(z)  # (B, N_CANDS, action_dim)
        action = policy.select_action_with_exploration(z)   # (B, action_dim)
        # Check that each batch element's action appears in its candidates
        for b in range(BATCH):
            dists = (cands[b] - action[b].unsqueeze(0)).norm(dim=-1)
            assert dists.min().item() < 1e-5




class TestUpdate:
    @pytest.fixture
    def batch(self) -> Dict[str, torch.Tensor]:
        torch.manual_seed(42)
        return {
            "log_probs": torch.randn(BATCH, HORIZON),
            "rewards":   torch.rand(BATCH, HORIZON),
        }

    def test_output_keys(self, policy, batch):
        out = policy.update(batch)
        assert set(out.keys()) == {"policy_loss"}

    def test_loss_is_scalar(self, policy, batch):
        out = policy.update(batch)
        assert out["policy_loss"].ndim == 0

    def test_loss_is_finite(self, policy, batch):
        out = policy.update(batch)
        assert torch.isfinite(out["policy_loss"])

    def test_gradient_flows_through_log_probs(self, policy, batch):
        log_probs = batch["log_probs"].requires_grad_(True)
        out = policy.update({**batch, "log_probs": log_probs})
        out["policy_loss"].backward()
        assert log_probs.grad is not None
        assert torch.isfinite(log_probs.grad).all()

    def test_positive_rewards_encourage_high_log_probs(self, policy):
        # With all-positive rewards, REINFORCE loss should decrease when
        # log_probs increase (gradient points in direction of higher log_probs)
        log_probs = torch.full((BATCH, HORIZON), -1.0, requires_grad=True)
        rewards   = torch.ones(BATCH, HORIZON)
        out = policy.update({"log_probs": log_probs, "rewards": rewards})
        out["policy_loss"].backward()
        # Gradient of loss w.r.t. log_probs should be negative
        # (increasing log_probs decreases loss)
        assert (log_probs.grad < 0).all()

    def test_zero_rewards_zero_loss(self, policy):
        batch = {
            "log_probs": torch.randn(BATCH, HORIZON),
            "rewards":   torch.zeros(BATCH, HORIZON),
        }
        out = policy.update(batch)
        assert out["policy_loss"].abs().item() < 1e-6
