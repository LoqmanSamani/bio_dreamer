from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from biodreamer.protein_dreamer.environment import (
    BaseOracle,
    BaseStructureOracle,
    CachedOracle,
    DMSLookupOracle,
    ESMFoldOracle,
    ESMFoldStructureOracle,
    PredictorOracle,
    ProteinEnvironment,
    TsuboyamaDMSOracle,
    WetLabOracle,
    _diff_to_mutation_list,
    _hamming,
    _sequence_to_mutant_string,
)



WT = "ACDEFGHIKL"   # 10-residue test sequence
MUT1 = "GCDEFGHIKL"  # position 0: A→G  ("A1G")
MUT2 = "ACDEFGHIKM"  # position 9: L→M  ("L10M")


def _simple_oracle(fitness: float = 1.0) -> "ConstOracle":
    return ConstOracle(fitness)


class ConstOracle(BaseOracle):
    """Always returns the same fitness score."""
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def query(self, sequence: str) -> float:
        return self.value


class DictOracle(BaseOracle):
    """Looks up fitness from a dict; NaN for unknown sequences."""
    def __init__(self, data: Dict[str, float]) -> None:
        self._data = data

    def query(self, sequence: str) -> float:
        return self._data.get(sequence, float("nan"))


class ConstStructureOracle(BaseStructureOracle):
    """Returns fixed zero-filled coords for any sequence."""
    def predict(self, sequence: str) -> Tuple[np.ndarray, np.ndarray, float]:
        L = len(sequence)
        coords = np.zeros((L, 3), dtype=np.float32)
        plddt = np.full(L, 80.0, dtype=np.float32)
        return coords, plddt, 0.9




class TestHelpers:
    def test_hamming_identical(self):
        assert _hamming("ACDE", "ACDE") == 0

    def test_hamming_all_different(self):
        assert _hamming("ACDE", "FGHI") == 4

    def test_hamming_partial(self):
        assert _hamming("ACDE", "ACDF") == 1

    def test_sequence_to_mutant_string_single(self):
        s = _sequence_to_mutant_string(MUT1, WT)
        assert s == "A1G"

    def test_sequence_to_mutant_string_double(self):
        seq = "GCDEFGHIKM"  # A→G at 0, L→M at 9
        s = _sequence_to_mutant_string(seq, WT)
        assert s == "A1G:L10M"

    def test_sequence_to_mutant_string_identical(self):
        assert _sequence_to_mutant_string(WT, WT) == ""

    def test_diff_to_mutation_list_single(self):
        lst = _diff_to_mutation_list(WT, MUT1)
        assert lst == ["A1G"]

    def test_diff_to_mutation_list_double(self):
        seq = "GCDEFGHIKM"
        lst = _diff_to_mutation_list(WT, seq)
        assert lst == ["A1G", "L10M"]

    def test_diff_to_mutation_list_identical(self):
        assert _diff_to_mutation_list(WT, WT) == []




class TestCachedOracle:
    def test_delegates_query(self):
        inner = ConstOracle(3.14)
        cached = CachedOracle(inner)
        assert cached.query(WT) == pytest.approx(3.14)

    def test_caches_on_second_call(self):
        call_count = []

        class CountingOracle(BaseOracle):
            def query(self, sequence):
                call_count.append(1)
                return 42.0

        cached = CachedOracle(CountingOracle())
        cached.query("ACDE")
        cached.query("ACDE")
        assert sum(call_count) == 1  # only called once

    def test_different_seqs_cached_separately(self):
        inner = DictOracle({"AAAA": 1.0, "BBBB": 2.0})
        cached = CachedOracle(inner)
        assert cached.query("AAAA") == pytest.approx(1.0)
        assert cached.query("BBBB") == pytest.approx(2.0)
        assert cached.cache_size == 2

    def test_clear_cache(self):
        call_count = []

        class CountingOracle(BaseOracle):
            def query(self, sequence):
                call_count.append(1)
                return 0.0

        cached = CachedOracle(CountingOracle())
        cached.query("X")
        cached.clear_cache()
        cached.query("X")
        assert sum(call_count) == 2

    def test_cache_size_property(self):
        cached = CachedOracle(ConstOracle(0.0))
        cached.query("A")
        cached.query("B")
        assert cached.cache_size == 2




class TestDMSLookupOracle:
    @pytest.fixture
    def oracle(self) -> DMSLookupOracle:
        data = {"A1G": 1.5, "L10M": -0.3, WT: 0.0}
        return DMSLookupOracle(data, wt_sequence=WT)

    def test_direct_key_hit(self, oracle):
        assert oracle.query("A1G") == pytest.approx(1.5)

    def test_direct_key_miss_returns_nan(self, oracle):
        assert math.isnan(oracle.query("Z99Q"))

    def test_wt_key_returns_zero(self, oracle):
        assert oracle.query(WT) == pytest.approx(0.0)

    def test_sequence_diff_lookup(self, oracle):
        # passing the full mutant sequence should resolve to "A1G"
        assert oracle.query(MUT1) == pytest.approx(1.5)

    def test_sequence_identical_to_wt(self, oracle):
        assert oracle.query(WT) == pytest.approx(0.0)

    def test_n_variants(self, oracle):
        assert oracle.n_variants == 3

    def test_normalize_shifts_values(self):
        data = {"A1G": 0.0, "L10M": 1.0, "K8R": 2.0}
        oracle = DMSLookupOracle(data, wt_sequence=WT, normalize=True)
        # wt (0.0) is added automatically, so normalization includes 4 values;
        # query all of them to verify the mean is zero
        all_keys = list(data.keys()) + [WT]
        vals = [oracle.query(k) for k in all_keys]
        mean = sum(vals) / len(vals)
        assert abs(mean) < 1e-5

    def test_dataframe_input(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        df = pd.DataFrame({"mutant": ["A1G", "L10M"], "DMS_score": [1.0, -0.5]})
        oracle = DMSLookupOracle(df, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(1.0)
        assert oracle.query("L10M") == pytest.approx(-0.5)

    def test_invalid_data_type_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DMSLookupOracle(12345, wt_sequence=WT)




class TestPredictorOracle:
    def test_callable_called(self):
        oracle = PredictorOracle(lambda seq: len(seq) * 0.1)
        assert oracle.query("AAAA") == pytest.approx(0.4)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            PredictorOracle("not_a_callable")  # type: ignore

    def test_exception_returns_nan(self):
        def bad(seq):
            raise RuntimeError("boom")

        oracle = PredictorOracle(bad)
        assert math.isnan(oracle.query("AAAA"))

    def test_result_is_float(self):
        oracle = PredictorOracle(lambda seq: 42)
        result = oracle.query("A")
        assert isinstance(result, float)




class TestWetLabOracle:
    def test_unknown_sequence_nan(self):
        oracle = WetLabOracle()
        assert math.isnan(oracle.query("UNKNOWN"))

    def test_register_and_query(self):
        oracle = WetLabOracle()
        oracle.register_measurement(WT, 2.5)
        assert oracle.query(WT) == pytest.approx(2.5)

    def test_register_batch(self):
        oracle = WetLabOracle()
        oracle.register_batch({WT: 1.0, MUT1: 1.8})
        assert oracle.query(WT) == pytest.approx(1.0)
        assert oracle.query(MUT1) == pytest.approx(1.8)

    def test_n_measured(self):
        oracle = WetLabOracle()
        oracle.register_measurement("A", 0.0)
        oracle.register_measurement("B", 0.0)
        assert oracle.n_measured == 2

    def test_overwrite_measurement(self):
        oracle = WetLabOracle()
        oracle.register_measurement(WT, 1.0)
        oracle.register_measurement(WT, 2.0)
        assert oracle.query(WT) == pytest.approx(2.0)




class TestESMFoldOracle:
    def test_construction(self):
        oracle = ESMFoldOracle(cache_dir=None, device=None)
        assert oracle.baseline_plddt is None

    def test_query_returns_nan_when_esmfold_unavailable(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("ESMFold not installed")

        monkeypatch.setattr(
            "biodreamer.protein_dreamer.data.preprocessing.predict_structure_esmfold",
            _raise,
        )
        oracle = ESMFoldOracle()
        result = oracle.query("ACDE")
        assert math.isnan(result)

    def test_query_uses_mean_plddt(self, monkeypatch):
        plddt = np.array([80.0, 90.0, 70.0])
        monkeypatch.setattr(
            "biodreamer.protein_dreamer.data.preprocessing.predict_structure_esmfold",
            lambda seq, **kw: (np.zeros((3, 3)), plddt, 0.9),
        )
        oracle = ESMFoldOracle()
        assert oracle.query("ACD") == pytest.approx(float(plddt.mean()))

    def test_query_subtracts_baseline(self, monkeypatch):
        monkeypatch.setattr(
            "biodreamer.protein_dreamer.data.preprocessing.predict_structure_esmfold",
            lambda seq, **kw: (np.zeros((3, 3)), np.array([80.0, 80.0, 80.0]), 0.9),
        )
        oracle = ESMFoldOracle(baseline_plddt=75.0)
        assert oracle.query("ACD") == pytest.approx(5.0)

    def test_set_baseline(self, monkeypatch):
        monkeypatch.setattr(
            "biodreamer.protein_dreamer.data.preprocessing.predict_structure_esmfold",
            lambda seq, **kw: (np.zeros((3, 3)), np.array([60.0, 80.0, 100.0]), 0.8),
        )
        oracle = ESMFoldOracle()
        baseline = oracle.set_baseline("ACD")
        assert baseline == pytest.approx(80.0)
        assert oracle.baseline_plddt == pytest.approx(80.0)




class TestESMFoldStructureOracle:
    def test_construction(self):
        oracle = ESMFoldStructureOracle(coord_mode="ca")
        assert oracle.coord_mode == "ca"

    def test_predict_delegates(self, monkeypatch):
        expected_coords = np.zeros((5, 3), dtype=np.float32)
        expected_plddt = np.full(5, 90.0, dtype=np.float32)
        monkeypatch.setattr(
            "biodreamer.protein_dreamer.data.preprocessing.predict_structure_esmfold",
            lambda seq, **kw: (expected_coords, expected_plddt, 0.95),
        )
        oracle = ESMFoldStructureOracle(coord_mode="ca")
        coords, plddt, ptm = oracle.predict("ACDEF")
        assert coords.shape == (5, 3)
        assert plddt is not None and len(plddt) == 5
        assert ptm == pytest.approx(0.95)




class TestProteinEnvironmentConstruction:
    def test_empty_wt_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ProteinEnvironment({}, "", ConstOracle())

    def test_invalid_oracle_raises(self):
        with pytest.raises(TypeError):
            ProteinEnvironment({}, WT, "not_an_oracle")  # type: ignore

    def test_no_reset_state_is_none(self):
        env = ProteinEnvironment({}, WT, ConstOracle())
        assert env.current_state is None

    def test_step_before_reset_raises(self):
        env = ProteinEnvironment({}, WT, ConstOracle())
        with pytest.raises(RuntimeError, match="reset"):
            env.step("A1G")

    def test_render_before_reset_raises(self):
        env = ProteinEnvironment({}, WT, ConstOracle())
        with pytest.raises(RuntimeError, match="reset"):
            env.render()




class TestProteinEnvironmentReset:
    @pytest.fixture
    def env(self) -> ProteinEnvironment:
        return ProteinEnvironment({}, WT, DictOracle({WT: 0.0}))

    def test_reset_returns_state_dict(self, env):
        state = env.reset()
        assert isinstance(state, dict)

    def test_reset_state_has_required_keys(self, env):
        state = env.reset()
        for key in ("sequence", "fitness", "coords", "plddt", "ptm"):
            assert key in state

    def test_reset_sequence_is_wt(self, env):
        state = env.reset()
        assert state["sequence"] == WT

    def test_reset_fitness_queried(self, env):
        state = env.reset()
        assert state["fitness"] == pytest.approx(0.0)

    def test_reset_coords_none_without_structure_oracle(self, env):
        state = env.reset()
        assert state["coords"] is None

    def test_reset_with_structure_oracle(self):
        env = ProteinEnvironment({}, WT, ConstOracle(), structure_oracle=ConstStructureOracle())
        state = env.reset()
        assert state["coords"] is not None
        assert state["coords"].shape == (len(WT), 3)

    def test_reset_clears_trajectory(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        env.reset()
        assert env.trajectory == []

    def test_reset_resets_step_count(self):
        env = ProteinEnvironment({}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        env.reset()
        _, _, _, info = env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        assert info["step"] == 1




class TestProteinEnvironmentStepString:
    @pytest.fixture
    def env(self) -> ProteinEnvironment:
        data = {WT: 0.0, MUT1: 1.5}
        return ProteinEnvironment({}, WT, DMSLookupOracle(data, WT))

    def test_step_returns_four_values(self, env):
        env.reset()
        result = env.step("A1G")
        assert len(result) == 4

    def test_step_next_state_has_mutant_sequence(self, env):
        env.reset()
        next_state, _, _, _ = env.step("A1G")
        assert next_state["sequence"] == MUT1

    def test_step_reward_is_oracle_fitness(self, env):
        env.reset()
        _, reward, _, _ = env.step("A1G")
        assert reward == pytest.approx(1.5)

    def test_step_info_contains_mutation_str(self, env):
        env.reset()
        _, _, _, info = env.step("A1G")
        assert info["mutation_str"] == "A1G"

    def test_step_info_hamming(self, env):
        env.reset()
        _, _, _, info = env.step("A1G")
        assert info["hamming"] == 1

    def test_step_info_step_counter(self, env):
        env.reset()
        env.step("A1G")
        _, _, _, info = env.step({"position": 9, "aa_old": 9, "aa_new": 11})
        assert info["step"] == 2

    def test_step_invalid_string_returns_done(self, env):
        env.reset()
        _, _, done, info = env.step("ZZZZ_INVALID")
        assert done
        assert "error" in info

    def test_step_current_state_updated(self, env):
        env.reset()
        env.step("A1G")
        assert env.current_state["sequence"] == MUT1




class TestProteinEnvironmentStepDict:
    @pytest.fixture
    def env(self) -> ProteinEnvironment:
        data = {WT: 0.0, MUT1: 2.0}
        return ProteinEnvironment({}, WT, DMSLookupOracle(data, WT))

    def test_dict_action_int_values(self, env):
        env.reset()
        # _AA_LIST = "ACDEFGHIKLMNPQRSTVWY" -> G is at index 5
        next_state, reward, done, info = env.step(
            {"position": 0, "aa_old": 0, "aa_new": 5}
        )
        assert next_state["sequence"][0] == "G"
        assert not math.isnan(reward)

    def test_dict_action_tensor_values(self, env):
        import torch
        env.reset()
        action = {
            "position": torch.tensor(0),
            "aa_old": torch.tensor(0),
            "aa_new": torch.tensor(5),  # G is at index 5
        }
        next_state, _, _, _ = env.step(action)
        assert next_state["sequence"][0] == "G"

    def test_dict_action_out_of_bounds_returns_done(self, env):
        env.reset()
        _, _, done, info = env.step({"position": 999, "aa_old": 0, "aa_new": 1})
        assert done
        assert "error" in info

    def test_dict_action_info_mutation_str_1based(self, env):
        env.reset()
        _, _, _, info = env.step({"position": 0, "aa_old": 0, "aa_new": 4})
        # position 0 (0-based) → "1" in mutation string
        assert info["mutation_str"][1] == "1"




class TestProteinEnvironmentDone:
    def test_done_on_max_steps(self):
        env = ProteinEnvironment({"max_steps": 2}, WT, ConstOracle(1.0))
        env.reset()
        _, _, done1, _ = env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        _, _, done2, _ = env.step({"position": 1, "aa_old": 2, "aa_new": 3})
        assert not done1
        assert done2

    def test_done_on_fitness_threshold(self):
        oracle = DictOracle({WT: 0.0, MUT1: 5.0})
        env = ProteinEnvironment({"fitness_threshold": 3.0}, WT, oracle)
        env.reset()
        _, _, done, _ = env.step("A1G")
        assert done

    def test_not_done_below_threshold(self):
        oracle = DictOracle({WT: 0.0, MUT1: 1.0})
        env = ProteinEnvironment({"fitness_threshold": 3.0, "max_steps": 20}, WT, oracle)
        env.reset()
        _, _, done, _ = env.step("A1G")
        assert not done

    def test_nan_reward_does_not_trigger_threshold(self):
        oracle = DictOracle({})  # always NaN
        env = ProteinEnvironment({"fitness_threshold": 0.0, "max_steps": 20}, WT, oracle)
        env.reset()
        _, _, done, _ = env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        assert not done




class TestProteinEnvironmentRender:
    @pytest.fixture
    def env(self) -> ProteinEnvironment:
        return ProteinEnvironment({}, WT, ConstOracle(1.0))

    def test_render_keys(self, env):
        env.reset()
        out = env.render()
        for key in ("alignment", "sequence", "wt_sequence", "mutations", "fitness", "step", "hamming"):
            assert key in out

    def test_render_alignment_all_match_at_reset(self, env):
        env.reset()
        out = env.render()
        assert set(out["alignment"]) <= {"|"}

    def test_render_alignment_after_mutation(self, env):
        env.reset()
        env.step("A1G")
        out = env.render()
        assert out["alignment"][0] == "X"
        assert all(c == "|" for c in out["alignment"][1:])

    def test_render_mutations_after_step(self, env):
        env.reset()
        env.step("A1G")
        out = env.render()
        assert "A1G" in out["mutations"]

    def test_render_hamming(self, env):
        env.reset()
        env.step("A1G")
        out = env.render()
        assert out["hamming"] == 1

    def test_render_step_counter(self, env):
        env.reset()
        env.step("A1G")
        out = env.render()
        assert out["step"] == 1




class TestProteinEnvironmentTrajectory:
    def test_trajectory_empty_after_reset(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        assert env.trajectory == []

    def test_trajectory_grows_with_steps(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        env.step({"position": 1, "aa_old": 2, "aa_new": 3})
        assert len(env.trajectory) == 2

    def test_trajectory_entry_keys(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        entry = env.trajectory[0]
        for key in ("action", "mutation_str", "state", "reward"):
            assert key in entry

    def test_trajectory_not_tracked_when_flag_false(self):
        env = ProteinEnvironment({}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        assert env.trajectory == []

    def test_trajectory_is_copy(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step({"position": 0, "aa_old": 0, "aa_new": 1})
        traj = env.trajectory
        traj.clear()
        assert len(env.trajectory) == 1  # internal list unaffected




class TestStateEncoderCompatibility:
    """Verify that states returned by reset/step carry the keys ProteinEncoder
    expects in embed_observation(): sequence, coords, plddt, ptm."""

    @pytest.fixture
    def env_with_struct(self) -> ProteinEnvironment:
        return ProteinEnvironment(
            {},
            WT,
            ConstOracle(1.0),
            structure_oracle=ConstStructureOracle(),
        )

    def test_reset_state_has_encoder_keys(self, env_with_struct):
        state = env_with_struct.reset()
        for key in ("sequence", "coords", "plddt", "ptm"):
            assert key in state

    def test_step_state_has_encoder_keys(self, env_with_struct):
        env_with_struct.reset()
        next_state, _, _, _ = env_with_struct.step("A1G")
        for key in ("sequence", "coords", "plddt", "ptm"):
            assert key in next_state

    def test_coords_shape(self, env_with_struct):
        state = env_with_struct.reset()
        assert state["coords"].shape == (len(WT), 3)

    def test_plddt_shape(self, env_with_struct):
        state = env_with_struct.reset()
        assert state["plddt"].shape == (len(WT),)

    def test_ptm_is_float(self, env_with_struct):
        state = env_with_struct.reset()
        assert isinstance(state["ptm"], float)




class TestDMSLookupOracleNegate:
    def test_negate_flips_sign(self):
        data = {"A1G": 1.0, "L10M": -0.5}
        oracle = DMSLookupOracle(data, wt_sequence=WT, negate=True)
        assert oracle.query("A1G") == pytest.approx(-1.0)
        assert oracle.query("L10M") == pytest.approx(0.5)

    def test_negate_false_preserves_sign(self):
        data = {"A1G": 1.0}
        oracle = DMSLookupOracle(data, wt_sequence=WT, negate=False)
        assert oracle.query("A1G") == pytest.approx(1.0)

    def test_negate_wt_baseline_becomes_zero_negated(self):
        # wt is added as 0.0; negating 0.0 is still 0.0
        oracle = DMSLookupOracle({}, wt_sequence=WT, negate=True)
        assert oracle.query(WT) == pytest.approx(0.0)

    def test_tsuboyama_ddg_column_detected(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        df = pd.DataFrame({"mutant": ["A1G", "L10M"], "ddG": [-0.5, 1.2]})
        oracle = DMSLookupOracle(df, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(-0.5)
        assert oracle.query("L10M") == pytest.approx(1.2)

    def test_tsuboyama_dg_column_detected(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        df = pd.DataFrame({"mutant": ["A1G"], "dG": [2.3]})
        oracle = DMSLookupOracle(df, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(2.3)




class TestTsuboyamaDMSOracle:
    def test_is_dms_lookup_oracle(self):
        oracle = TsuboyamaDMSOracle({"A1G": -1.0}, wt_sequence=WT)
        assert isinstance(oracle, DMSLookupOracle)

    def test_negates_by_default(self):
        # ΔΔG = -1.0 kcal/mol (stabilising) → reward = +1.0
        oracle = TsuboyamaDMSOracle({"A1G": -1.0}, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(1.0)

    def test_positive_ddg_destabilising_becomes_negative_reward(self):
        # ΔΔG = +2.0 (destabilising) → reward = -2.0
        oracle = TsuboyamaDMSOracle({"A1G": 2.0}, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(-2.0)

    def test_negate_false_returns_raw(self):
        oracle = TsuboyamaDMSOracle({"A1G": -1.0}, wt_sequence=WT, negate=False)
        assert oracle.query("A1G") == pytest.approx(-1.0)

    def test_sequence_diff_lookup(self):
        oracle = TsuboyamaDMSOracle({"A1G": -0.8}, wt_sequence=WT)
        # full mutant sequence lookup — should negate -0.8 → 0.8
        assert oracle.query(MUT1) == pytest.approx(0.8)

    def test_nan_for_unmeasured_variant(self):
        oracle = TsuboyamaDMSOracle({"A1G": -1.0}, wt_sequence=WT)
        assert math.isnan(oracle.query("L10M"))

    def test_dataframe_with_ddg_column(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        df = pd.DataFrame({"mutant": ["A1G", "L10M"], "ddG": [-0.5, 1.2]})
        oracle = TsuboyamaDMSOracle(df, wt_sequence=WT)
        assert oracle.query("A1G") == pytest.approx(0.5)   # negated
        assert oracle.query("L10M") == pytest.approx(-1.2)  # negated




# WT = "ACDEFGHIKL" — 10 residues
# MUT_DOUBLE = "GCDEFGHIKM"  positions 0 (A→G) and 9 (L→M)
MUT_DOUBLE = "GCDEFGHIKM"


class TestProteinEnvironmentMultiSite:
    @pytest.fixture
    def env(self) -> ProteinEnvironment:
        data = {
            WT: 0.0,
            "A1G:L10M": 2.5,   # double mutant string key
            MUT_DOUBLE: 2.5,   # also reachable by sequence diff
        }
        return ProteinEnvironment({}, WT, DMSLookupOracle(data, WT))

    def test_double_mutant_string_applies_both_sites(self, env):
        env.reset()
        next_state, _, _, _ = env.step("A1G:L10M")
        seq = next_state["sequence"]
        assert seq[0] == "G"   # A→G at position 0
        assert seq[9] == "M"   # L→M at position 9

    def test_double_mutant_reward_from_oracle(self, env):
        env.reset()
        _, reward, _, _ = env.step("A1G:L10M")
        assert reward == pytest.approx(2.5)

    def test_info_n_mutations_double(self, env):
        env.reset()
        _, _, _, info = env.step("A1G:L10M")
        assert info["n_mutations"] == 2

    def test_info_hamming_double(self, env):
        env.reset()
        _, _, _, info = env.step("A1G:L10M")
        assert info["hamming"] == 2

    def test_info_n_mutations_single(self, env):
        env.reset()
        _, _, _, info = env.step("A1G")
        assert info["n_mutations"] == 1

    def test_single_site_unaffected(self, env):
        env.reset()
        next_state, _, _, _ = env.step("A1G")
        assert next_state["sequence"][0] == "G"
        assert next_state["sequence"][9] == "L"  # L unchanged

    def test_tsuboyama_oracle_with_double_mutation(self):
        data = {"A1G:L10M": -1.5}
        oracle = TsuboyamaDMSOracle(data, wt_sequence=WT)
        env = ProteinEnvironment({}, WT, oracle)
        env.reset()
        _, reward, _, _ = env.step("A1G:L10M")
        assert reward == pytest.approx(1.5)  # negated by TsuboyamaDMSOracle

    def test_trajectory_stores_list_for_multi_site(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step("A1G:L10M")
        action_stored = env.trajectory[0]["action"]
        assert isinstance(action_stored, list)
        assert len(action_stored) == 2

    def test_trajectory_stores_dict_for_single_site(self):
        env = ProteinEnvironment({"track_trajectory": True}, WT, ConstOracle())
        env.reset()
        env.step("A1G")
        action_stored = env.trajectory[0]["action"]
        assert isinstance(action_stored, dict)
