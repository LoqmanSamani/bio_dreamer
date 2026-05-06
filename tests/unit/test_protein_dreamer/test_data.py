from __future__ import annotations

import math
from io import StringIO
from typing import Dict

import pandas as pd
import pytest
import torch

from biodreamer.protein_dreamer.data.dataset import (
    AssayType,
    CustomAssayDataset,
    FitnessTransitionDataset,
    ProteinGymDataset,
    TsuboyamaDataset,
    _make_targets,
    build_assay_type_map,
)
from biodreamer.protein_dreamer.data.loaders import make_collate_fn, make_dataloader
from biodreamer.protein_dreamer.data.preprocessing import (
    encode_mutation,
    normalize_fitness,
    parse_mutation_string,
)


_WT = "ACDEFGHIKLMNPQRSTVWY"  # 20-residue synthetic wild-type

def _simple_df(
    wt: str = _WT,
    mutation: str = "A1C",
    fitness: float = 0.75,
    assay_id: str = "TEST_ASSAY",
) -> pd.DataFrame:
    mutant = "C" + wt[1:]
    return pd.DataFrame(
        [{"wt_sequence": wt, "mutation": mutation, "mutant_sequence": mutant,
          "score": fitness, "assay_id": assay_id}]
    )


def _multi_df() -> pd.DataFrame:
    rows = [
        {"wt_sequence": _WT, "mutation": "A1C", "mutant_sequence": "C" + _WT[1:],
         "score": 0.5, "assay_id": "STAB_001"},
        {"wt_sequence": _WT, "mutation": "C2D", "mutant_sequence": _WT[0] + "D" + _WT[2:],
         "score": 0.8, "assay_id": "BIND_001"},
        {"wt_sequence": _WT, "mutation": "D3E", "mutant_sequence": _WT[:2] + "E" + _WT[3:],
         "score": 0.3, "assay_id": "ACTV_001"},
    ]
    return pd.DataFrame(rows)


_ASSAY_TYPE_MAP = {
    "STAB_001": AssayType.STABILITY,
    "BIND_001": AssayType.BINDING_AFFINITY,
    "ACTV_001": AssayType.CATALYTIC_ACTIVITY,
}


class TestParseMutationString:
    def test_single_substitution(self):
        parsed = parse_mutation_string("A42G")
        assert len(parsed.records) == 1
        rec = parsed.records[0]
        assert rec.wt_aa == "A"
        assert rec.position == 42
        assert rec.mut_aa == "G"
        assert rec.position_0 == 41

    def test_multi_site(self):
        parsed = parse_mutation_string("A42G:K56R")
        assert len(parsed.records) == 2
        assert parsed.records[0].position == 42
        assert parsed.records[1].position == 56

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            parse_mutation_string("not_a_mutation!!")

    def test_is_single_property(self):
        assert parse_mutation_string("A1C").is_single
        assert not parse_mutation_string("A1C:K2R").is_single


class TestAssayTypeMap:
    def test_build_from_csv(self, tmp_path):
        csv = tmp_path / "ref.csv"
        csv.write_text(
            "DMS_id,selection_type\n"
            "PROT_001,Stability\n"
            "PROT_002,Binding\n"
            "PROT_003,Activity\n"
            "PROT_004,OrganismalFitness\n"
        )
        mapping = build_assay_type_map(str(csv))
        assert mapping["PROT_001"] == AssayType.STABILITY
        assert mapping["PROT_002"] == AssayType.BINDING_AFFINITY
        assert mapping["PROT_003"] == AssayType.CATALYTIC_ACTIVITY
        assert mapping["PROT_004"] == AssayType.ORGANISMAL_FITNESS

    def test_missing_type_column_defaults_to_organismal(self, tmp_path):
        csv = tmp_path / "ref_no_type.csv"
        csv.write_text("DMS_id\nPROT_001\n")
        mapping = build_assay_type_map(str(csv))
        assert mapping["PROT_001"] == AssayType.ORGANISMAL_FITNESS

    def test_make_targets_stability(self):
        t = _make_targets(1.0, AssayType.STABILITY)
        assert t["stability"] == 1.0
        assert math.isnan(t["affinity"])
        assert math.isnan(t["activity"])

    def test_make_targets_binding(self):
        t = _make_targets(0.5, AssayType.BINDING_AFFINITY)
        assert math.isnan(t["stability"])
        assert t["affinity"] == 0.5
        assert math.isnan(t["activity"])

    def test_make_targets_activity(self):
        t = _make_targets(0.3, AssayType.CATALYTIC_ACTIVITY)
        assert math.isnan(t["stability"])
        assert math.isnan(t["affinity"])
        assert t["activity"] == 0.3

    def test_make_targets_organismal_routes_to_fitness(self):
        t = _make_targets(0.9, AssayType.ORGANISMAL_FITNESS)
        assert t["fitness"] == 0.9
        assert math.isnan(t["stability"])
        assert math.isnan(t["affinity"])
        assert math.isnan(t["activity"])

    def test_make_targets_none_fitness(self):
        t = _make_targets(None, AssayType.STABILITY)
        assert all(math.isnan(v) for v in t.values())



class TestProteinGymDataset:
    def test_basic_length(self):
        ds = ProteinGymDataset({}, _simple_df())
        assert len(ds) == 1

    def test_item_keys(self):
        ds = ProteinGymDataset({}, _simple_df())
        item = ds[0]
        assert "wt_sequence" in item
        assert "mutant_sequence" in item
        assert "targets" in item
        assert "action" in item

    def test_targets_dict_structure(self):
        ds = ProteinGymDataset({}, _simple_df())
        item = ds[0]
        assert set(item["targets"].keys()) == {"stability", "affinity", "activity", "fitness"}

    def test_default_routes_fitness_to_organismal_fitness(self):
        ds = ProteinGymDataset({}, _simple_df(fitness=0.75))
        item = ds[0]
        assert item["targets"]["fitness"] == pytest.approx(0.75)
        assert math.isnan(item["targets"]["stability"])
        assert math.isnan(item["targets"]["affinity"])
        assert math.isnan(item["targets"]["activity"])

    def test_assay_type_map_routing(self):
        df = _multi_df()
        ds = ProteinGymDataset({}, df, assay_type_map=_ASSAY_TYPE_MAP)
        items = [ds[i] for i in range(len(ds))]
        stab = next(it for it in items if it["assay_id"] == "STAB_001")
        bind = next(it for it in items if it["assay_id"] == "BIND_001")
        actv = next(it for it in items if it["assay_id"] == "ACTV_001")

        assert not math.isnan(stab["targets"]["stability"])
        assert math.isnan(stab["targets"]["affinity"])

        assert not math.isnan(bind["targets"]["affinity"])
        assert math.isnan(bind["targets"]["stability"])

        assert not math.isnan(actv["targets"]["activity"])
        assert math.isnan(actv["targets"]["stability"])

    def test_action_is_dict_with_position(self):
        ds = ProteinGymDataset({}, _simple_df())
        action = ds[0]["action"]
        assert action is not None
        assert "position" in action
        assert "aa_old" in action
        assert "aa_new" in action

    def test_action_position_is_long_tensor(self):
        ds = ProteinGymDataset({}, _simple_df(mutation="A1C"))
        action = ds[0]["action"]
        assert action["position"].dtype == torch.long
        assert action["position"].item() == 0  # 0-based

    def test_dataframe_input(self):
        df = _simple_df()
        ds = ProteinGymDataset({}, df)
        assert len(ds) == 1

    def test_missing_fitness_gives_nan_targets(self):
        df = pd.DataFrame([{
            "wt_sequence": _WT,
            "mutation": "A1C",
            "mutant_sequence": "C" + _WT[1:],
        }])
        ds = ProteinGymDataset({}, df)
        item = ds[0]
        assert all(math.isnan(v) for v in item["targets"].values())



class TestTsuboyamaDataset:
    def test_all_targets_route_to_stability(self):
        df = _multi_df()
        # even with an affinity assay_type_map, TsuboyamaDataset overrides to stability
        ds = TsuboyamaDataset({}, df, assay_type_map=_ASSAY_TYPE_MAP)
        for i in range(len(ds)):
            item = ds[i]
            assert not math.isnan(item["targets"]["stability"])
            assert math.isnan(item["targets"]["affinity"])
            assert math.isnan(item["targets"]["activity"])



class TestFitnessTransitionDataset:
    def test_structure(self):
        base = ProteinGymDataset({}, _simple_df(fitness=0.75))
        td = FitnessTransitionDataset(base)
        assert len(td) == 1
        item = td[0]
        assert "s_t" in item
        assert "s_t1" in item
        assert "action" in item
        assert "reward" in item
        assert "targets" in item

    def test_reward_is_primary_target(self):
        base = ProteinGymDataset({}, _simple_df(fitness=0.75))
        td = FitnessTransitionDataset(base)
        assert td[0]["reward"] == pytest.approx(0.75)

    def test_targets_propagated(self):
        df = _multi_df()
        base = ProteinGymDataset({}, df, assay_type_map=_ASSAY_TYPE_MAP)
        td = FitnessTransitionDataset(base)
        for i in range(len(td)):
            item = td[i]
            targets = item["targets"]
            assert set(targets.keys()) == {"stability", "affinity", "activity", "fitness"}

    def test_sequence_in_states(self):
        base = ProteinGymDataset({}, _simple_df())
        td = FitnessTransitionDataset(base)
        item = td[0]
        assert isinstance(item["s_t"]["sequence"], str)
        assert isinstance(item["s_t1"]["sequence"], str)
        assert item["s_t"]["sequence"] != item["s_t1"]["sequence"]

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            FitnessTransitionDataset("not_a_dataset")  # type: ignore



class TestCollation:
    def _make_batch(self):
        base = ProteinGymDataset({}, _multi_df(), assay_type_map=_ASSAY_TYPE_MAP)
        td = FitnessTransitionDataset(base)
        return [td[i] for i in range(len(td))]

    def test_targets_stacked_to_tensors(self):
        batch = self._make_batch()
        collate = make_collate_fn()
        out = collate(batch)
        assert "targets" in out
        targets = out["targets"]
        assert isinstance(targets, dict)
        assert set(targets.keys()) == {"stability", "affinity", "activity", "fitness"}
        for v in targets.values():
            assert isinstance(v, torch.Tensor)
            assert v.shape == (3,)  # batch size 3
            assert v.dtype == torch.float32

    def test_nan_preserved_in_targets(self):
        batch = self._make_batch()
        collate = make_collate_fn()
        out = collate(batch)
        targets = out["targets"]
        # STAB row: affinity, activity, fitness should be NaN
        assert torch.isnan(targets["affinity"][0])
        assert torch.isnan(targets["activity"][0])
        assert torch.isnan(targets["fitness"][0])
        # BIND row: stability, activity, fitness should be NaN
        assert torch.isnan(targets["stability"][1])
        assert torch.isnan(targets["activity"][1])
        assert torch.isnan(targets["fitness"][1])
        # ACTV row: stability, affinity, fitness should be NaN
        assert torch.isnan(targets["stability"][2])
        assert torch.isnan(targets["affinity"][2])
        assert torch.isnan(targets["fitness"][2])

    def test_empty_batch(self):
        collate = make_collate_fn()
        out = collate([])
        assert out == {}


class TestNormalizeFitness:
    def test_minmax(self):
        result = normalize_fitness([0.0, 0.5, 1.0], method="minmax")
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_zscore(self):
        result = normalize_fitness([0.0, 0.5, 1.0], method="zscore")
        assert abs(result.mean()) < 1e-6

    def test_quantile(self):
        result = normalize_fitness([3.0, 1.0, 2.0], method="quantile")
        assert result[1] == pytest.approx(0.0)   # min rank
        assert result[0] == pytest.approx(1.0)   # max rank

    def test_single_value(self):
        result = normalize_fitness([5.0], method="quantile")
        assert result[0] == pytest.approx(0.0)


class TestEncodeMutation:
    def test_position_is_int64(self):
        action = encode_mutation("A1C")
        assert action["position"].dtype == torch.long

    def test_position_is_zero_based(self):
        action = encode_mutation("A1C")
        assert action["position"].item() == 0

    def test_aa_indices_are_long(self):
        action = encode_mutation("A1C")
        assert action["aa_old"].dtype == torch.long
        assert action["aa_new"].dtype == torch.long
