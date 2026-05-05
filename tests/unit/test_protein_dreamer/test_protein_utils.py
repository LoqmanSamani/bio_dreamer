"""Unit tests for biodreamer.protein_dreamer.utils."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from biodreamer.protein_dreamer.utils import (
    EMAUpdater,
    SIGReg,
    _AA_LIST,
    _AA_TO_IDX,
    aa_indices_to_sequence,
    compute_hamming_distance,
    normalize_rewards,
    sequence_to_aa_indices,
)






class TestSIGReg:
    @pytest.fixture
    def reg(self) -> SIGReg:
        return SIGReg({"latent_dim": 32, "num_proj": 8})

    @pytest.fixture
    def z_gaussian(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(128, 32)

    @pytest.fixture
    def z_small(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(4, 32)

    def test_output_is_scalar(self, reg, z_gaussian):
        loss = reg(z_gaussian)
        assert loss.ndim == 0

    def test_output_non_negative(self, reg, z_gaussian):
        assert reg(z_gaussian).item() >= 0.0

    def test_output_finite(self, reg, z_gaussian):
        assert torch.isfinite(reg(z_gaussian))

    def test_gradient_flows(self, reg, z_gaussian):
        z = z_gaussian.requires_grad_(True)
        reg(z).backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_3d_input_flattened(self, reg):
        z = torch.randn(4, 8, 32)
        loss = reg(z)
        assert loss.ndim == 0

    def test_gaussian_input_small_loss(self, reg, z_gaussian):
        # well-normalised embeddings should give a small loss
        loss = reg(z_gaussian).item()
        assert loss < 2.0

    def test_shifted_input_larger_loss(self, reg, z_gaussian):
        loss_clean = reg(z_gaussian).item()
        z_shifted = z_gaussian + 5.0
        loss_shifted = reg(z_shifted).item()
        assert loss_shifted > loss_clean

    def test_scaled_input_larger_loss(self, reg, z_gaussian):
        loss_clean = reg(z_gaussian).item()
        z_scaled = z_gaussian * 5.0
        loss_scaled = reg(z_scaled).item()
        assert loss_scaled > loss_clean

    def test_mean_weight_zero_removes_mean_term(self):
        reg_no_mean = SIGReg({"latent_dim": 32, "num_proj": 8, "mean_weight": 0.0,
                               "var_weight": 0.0, "proj_weight": 0.0})
        z = torch.randn(64, 32) + 10.0
        assert reg_no_mean(z).item() == pytest.approx(0.0, abs=1e-6)

    def test_single_sample_does_not_nan(self, reg):
        z = torch.randn(1, 32)
        loss = reg(z)
        assert not torch.isnan(loss)

    def test_device_inferred_from_input(self, reg):
        # move input to CPU explicitly — no error, no self.device mismatch
        z = torch.randn(16, 32, device="cpu")
        loss = reg(z)
        assert loss.device.type == "cpu"

    def test_no_learnable_parameters(self, reg):
        assert sum(1 for _ in reg.parameters()) == 0





class TestEMAUpdater:
    def _make_linear(self, fill: float) -> nn.Linear:
        m = nn.Linear(4, 4, bias=False)
        nn.init.constant_(m.weight, fill)
        return m

    def test_invalid_tau_zero_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            EMAUpdater({"tau": 0.0})

    def test_invalid_tau_one_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            EMAUpdater({"tau": 1.0})

    def test_invalid_tau_negative_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            EMAUpdater({"tau": -0.5})

    def test_tau_stored(self):
        ema = EMAUpdater({"tau": 0.95})
        assert ema.tau == pytest.approx(0.95)

    def test_tau_close_to_one_barely_changes_target(self):
        ema = EMAUpdater({"tau": 0.999})
        online = self._make_linear(1.0)
        target = self._make_linear(0.0)
        ema.update(online, target)
        # target should be close to 0 still (moved only 0.001 toward online)
        w = target.weight.data
        assert w.mean().item() == pytest.approx(0.001, abs=1e-6)

    def test_tau_close_to_zero_copies_online(self):
        ema = EMAUpdater({"tau": 1e-9})
        online = self._make_linear(7.0)
        target = self._make_linear(0.0)
        ema.update(online, target)
        assert torch.allclose(target.weight.data, online.weight.data, atol=1e-5)

    def test_correct_ema_formula(self):
        tau = 0.9
        ema = EMAUpdater({"tau": tau})
        online = self._make_linear(1.0)
        target = self._make_linear(0.0)
        ema.update(online, target)
        expected = tau * 0.0 + (1 - tau) * 1.0
        assert target.weight.data.mean().item() == pytest.approx(expected, abs=1e-6)

    def test_repeated_updates_converge(self):
        ema = EMAUpdater({"tau": 0.9})
        online = self._make_linear(1.0)
        target = self._make_linear(0.0)
        for _ in range(200):
            ema.update(online, target)
        # after many updates, target should be very close to online
        assert torch.allclose(target.weight.data, online.weight.data, atol=1e-3)

    def test_online_parameters_not_modified(self):
        ema = EMAUpdater({"tau": 0.9})
        online = self._make_linear(3.0)
        target = self._make_linear(0.0)
        original = online.weight.data.clone()
        ema.update(online, target)
        assert torch.equal(online.weight.data, original)

    def test_no_grad_update(self):
        ema = EMAUpdater({"tau": 0.9})
        online = self._make_linear(1.0)
        target = self._make_linear(0.0)
        ema.update(online, target)
        assert target.weight.grad is None





class TestSequenceToAAIndices:
    def test_output_dtype(self):
        idx = sequence_to_aa_indices("ACDEF")
        assert idx.dtype == torch.long

    def test_output_length(self):
        seq = "ACDEFGHIKLM"
        assert sequence_to_aa_indices(seq).shape == (len(seq),)

    def test_known_amino_acids(self):
        idx = sequence_to_aa_indices("AC")
        assert idx[0].item() == _AA_TO_IDX["A"]
        assert idx[1].item() == _AA_TO_IDX["C"]

    def test_all_canonical_aas(self):
        seq = "".join(_AA_LIST)
        idx = sequence_to_aa_indices(seq)
        assert (idx >= 0).all()
        assert (idx < 20).all()

    def test_lowercase_normalised(self):
        lower = sequence_to_aa_indices("acdef")
        upper = sequence_to_aa_indices("ACDEF")
        assert torch.equal(lower, upper)

    def test_unknown_character_maps_to_minus_one(self):
        idx = sequence_to_aa_indices("A1C")
        assert idx[1].item() == -1

    def test_empty_sequence(self):
        idx = sequence_to_aa_indices("")
        assert idx.shape == (0,)

    def test_indices_in_range_for_valid_sequence(self):
        idx = sequence_to_aa_indices("ACDEFGHIKLMNPQRSTVWY")
        assert (idx >= 0).all() and (idx < 20).all()

    def test_unique_index_per_aa(self):
        seq = "".join(_AA_LIST)
        idx = sequence_to_aa_indices(seq)
        assert len(set(idx.tolist())) == 20





class TestAAIndicesToSequence:
    def test_roundtrip(self):
        seq = "ACDEFGHIK"
        recovered = aa_indices_to_sequence(sequence_to_aa_indices(seq))
        assert recovered == seq

    def test_full_alphabet_roundtrip(self):
        seq = "".join(_AA_LIST)
        assert aa_indices_to_sequence(sequence_to_aa_indices(seq)) == seq

    def test_unknown_index_maps_to_question_mark(self):
        idx = torch.tensor([-1, 0, 25], dtype=torch.long)
        result = aa_indices_to_sequence(idx)
        assert result[0] == "?"
        assert result[2] == "?"

    def test_output_length(self):
        idx = torch.arange(5, dtype=torch.long)
        assert len(aa_indices_to_sequence(idx)) == 5





class TestComputeHammingDistance:
    def test_identical_sequences(self):
        assert compute_hamming_distance("ACDEF", "ACDEF") == 0

    def test_all_different(self):
        assert compute_hamming_distance("AAAAA", "CCCCC") == 5

    def test_one_mutation(self):
        assert compute_hamming_distance("ACDEF", "GCDEF") == 1

    def test_two_mutations(self):
        assert compute_hamming_distance("ACDEF", "GCDEF"[:1] + "D" + "DEF") == 2

    def test_empty_sequences(self):
        assert compute_hamming_distance("", "") == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_hamming_distance("ACDE", "ACDEF")

    def test_case_sensitive(self):
        # uppercase A ≠ lowercase a
        assert compute_hamming_distance("A", "a") == 1

    def test_single_character_match(self):
        assert compute_hamming_distance("A", "A") == 0

    def test_single_character_mismatch(self):
        assert compute_hamming_distance("A", "C") == 1





class TestNormalizeRewards:
    def test_output_shape_preserved(self):
        r = torch.randn(4, 8)
        assert normalize_rewards(r).shape == (4, 8)

    def test_mean_approximately_zero(self):
        r = torch.randn(256)
        normed = normalize_rewards(r)
        assert normed.mean().abs().item() < 1e-5

    def test_std_approximately_one(self):
        r = torch.randn(256)
        normed = normalize_rewards(r)
        assert normed.std().item() == pytest.approx(1.0, abs=1e-2)

    def test_constant_rewards_no_nan(self):
        r = torch.ones(16) * 5.0
        normed = normalize_rewards(r)
        assert not torch.isnan(normed).any()

    def test_constant_rewards_all_zero(self):
        # constant rewards normalise to ~0 (std≈0, eps prevents inf)
        r = torch.ones(16) * 5.0
        normed = normalize_rewards(r, eps=1e-8)
        assert normed.abs().max().item() < 1e-3

    def test_output_finite_for_normal_input(self):
        r = torch.randn(64)
        assert torch.isfinite(normalize_rewards(r)).all()

    def test_gradient_flows(self):
        r = torch.randn(16, requires_grad=True)
        normalize_rewards(r).sum().backward()
        assert r.grad is not None

    def test_single_element_no_nan(self):
        r = torch.tensor([3.0])
        normed = normalize_rewards(r)
        assert not torch.isnan(normed).any()

    def test_custom_eps(self):
        r = torch.ones(8)
        normed = normalize_rewards(r, eps=1.0)
        assert torch.isfinite(normed).all()
