from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore





class AssayType(Enum):
    STABILITY          = "stability"
    BINDING_AFFINITY   = "affinity"
    CATALYTIC_ACTIVITY = "activity"
    ORGANISMAL_FITNESS = "fitness"  # catch-all for growth/survival/phenotypic assays


_SELECTION_TYPE_MAP: Dict[str, AssayType] = {
    # stability
    "stability":        AssayType.STABILITY,
    "thermostability":  AssayType.STABILITY,
    "folding":          AssayType.STABILITY,
    "expression":       AssayType.STABILITY,  # expression assays are proxies for foldability
    # binding affinity
    "binding":          AssayType.BINDING_AFFINITY,
    "affinity":         AssayType.BINDING_AFFINITY,
    "interaction":      AssayType.BINDING_AFFINITY,
    # catalytic activity
    "activity":         AssayType.CATALYTIC_ACTIVITY,
    "catalytic":        AssayType.CATALYTIC_ACTIVITY,
    "enzyme":           AssayType.CATALYTIC_ACTIVITY,
    "kcat":             AssayType.CATALYTIC_ACTIVITY,
    # organismal fitness (growth, survival, competitive selection)
    "organismal":       AssayType.ORGANISMAL_FITNESS,
    "growth":           AssayType.ORGANISMAL_FITNESS,
    "fitness":          AssayType.ORGANISMAL_FITNESS,
    "survival":         AssayType.ORGANISMAL_FITNESS,
    "replication":      AssayType.ORGANISMAL_FITNESS,
}


def build_assay_type_map(reference_csv_path: str) -> Dict[str, AssayType]:
    """parse a ProteinGym reference csv and return a dms_id -> AssayType mapping"""
    if pd is None:
        raise RuntimeError("'pandas' package is required to parse the ProteinGym reference CSV")

    df = pd.read_csv(reference_csv_path)
    id_col = _find_column(df, ["DMS_id", "assay_id", "id"])
    type_col = _find_column(df, ["selection_type", "assay_type", "type"])

    if id_col is None:
        raise ValueError(
            f"Cannot find DMS_id column in {reference_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    if type_col is None:
        logger.warning(
            "No selection_type column found in %s — all assays default to ORGANISMAL_FITNESS",
            reference_csv_path,
        )
        return {str(row[id_col]): AssayType.ORGANISMAL_FITNESS for _, row in df.iterrows()}

    result: Dict[str, AssayType] = {}
    for _, row in df.iterrows():
        dms_id = str(row[id_col])
        raw_type = str(row[type_col]).strip().lower() if pd.notna(row[type_col]) else ""
        assay_type = AssayType.ORGANISMAL_FITNESS
        for keyword, mapped_type in _SELECTION_TYPE_MAP.items():
            if keyword in raw_type:
                assay_type = mapped_type
                break
        result[dms_id] = assay_type

    return result


def _make_targets(fitness: Optional[float], assay_type: AssayType) -> Dict[str, float]:
    """route a fitness measurement to the correct target key based on assay type.
    
    the four target keys map to the four reward heads:
        - stability: ΔΔG / ΔG / folding measurements (Tsuboyama, ProteinGym stability)
        - affinity: binding affinity / Kd (ProteinGym binding assays)
        - activity:  catalytic activity / kcat (ProteinGym activity assays)
        - fitness:   organismal/holistic fitness (ProteinGym OrganismalFitness assays,
                      unknown assay types). Acts as the catch-all head.
    """
    nan = float("nan")
    targets: Dict[str, float] = {
        "stability": nan,
        "affinity":  nan,
        "activity":  nan,
        "fitness":   nan,
    }
    if fitness is None:
        return targets
    if assay_type == AssayType.STABILITY:
        targets["stability"] = fitness
    elif assay_type == AssayType.BINDING_AFFINITY:
        targets["affinity"] = fitness
    elif assay_type == AssayType.CATALYTIC_ACTIVITY:
        targets["activity"] = fitness
    else:  # ORGANISMAL_FITNESS and any unrecognised type
        targets["fitness"] = fitness
    return targets


def _find_column(df: "pd.DataFrame", candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None



class ProteinGymDataset(Dataset):
    """loads ProteinGym-style dms tables (parquet or csv).

    each item returns a dict with keys:
      - `wt_sequence`, `mutation_string`, `mutant_sequence`
      - `parsed_mutation`, `action`
      - `targets`: dict with keys `stability`, `affinity`, `activity`, `fitness`
        (only one is non-NaN per sample, depending on assay_type_map)
      - `assay_id`

    optionally also returns `input_wt` and `input_mutant` (tokenized) if
    a tokenizer is supplied.
    """
    SEQ_CANDIDATES = [
        "wt_sequence", "wild_type_sequence", "wildtype",
        "sequence", "wt_seq", "reference_sequence",
    ]
    MUT_CANDIDATES = [
        "mutation", "mutation_string", "mutations",
        "variant", "variant_str", "mutant", "mutant_sequence", "mutated_sequence",
    ]
    FITNESS_CANDIDATES = [
        "score", "fitness", "measurement", "value", "ddG", "dG", "enrichment",
    ]
    ASSAY_CANDIDATES = ["assay_id", "assay", "dataset", "protein", "id", "protein_id"]

    def __init__(
        self,
        config: dict,
        path: "str | pd.DataFrame",
        tokenizer: Optional[Any] = None,
        seq_col: Optional[str] = None,
        mutation_col: Optional[str] = None,
        mutant_col: Optional[str] = None,
        fitness_col: Optional[str] = None,
        assay_col: Optional[str] = None,
        assay_type_map: Optional[Dict[str, AssayType]] = None,
    ) -> None:
        max_length       = config.get("max_length", None)
        return_tensors   = config.get("return_tensors", False)
        strict_wt_check  = config.get("strict_wt_check", True)
        load_structures  = config.get("load_structures", False)
        struct_cache_dir = config.get("struct_cache_dir", None)
        tokenizer_mode   = config.get("tokenizer_mode", "char")
        pretokenize      = config.get("pretokenize", True)
        if pd is None:
            raise RuntimeError("pandas is required to load ProteinGym datasets")

        if isinstance(path, pd.DataFrame):
            df = path.copy()
        else:
            p = str(path)
            if p.endswith(".parquet") or p.endswith(".parq"):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)

        # resolve column names once at init
        seq_col     = seq_col     or _find_column(df, self.SEQ_CANDIDATES)
        mutation_col = mutation_col or _find_column(df, self.MUT_CANDIDATES)
        mutant_col  = mutant_col  or _find_column(
            df, ["mutant_sequence", "mutated_sequence", "mutant_seq"]
        ) or mutation_col
        fitness_col  = fitness_col  or _find_column(df, self.FITNESS_CANDIDATES)
        assay_col    = assay_col    or _find_column(df, self.ASSAY_CANDIDATES)

        if seq_col is None:
            raise RuntimeError(
                "Could not infer wild-type sequence column. "
                f"Available columns: {list(df.columns)}"
            )

        self.tokenizer       = tokenizer
        self.max_length      = max_length
        self.return_tensors  = return_tensors
        self.strict_wt_check = strict_wt_check
        self.load_structures = load_structures
        self.struct_cache_dir= struct_cache_dir
        self.tokenizer_mode  = tokenizer_mode
        self.assay_type_map  = assay_type_map or {}

        from .preprocessing import parse_mutation_string, apply_mutations, encode_mutation

        self._records: List[Dict[str, Any]] = []
        
        for _, row in df.iterrows():
            wt_seq   = str(row[seq_col]) if pd.notna(row[seq_col]) else None
            mut_str  = None
            if mutation_col and pd.notna(row.get(mutation_col, None)):
                mut_str = str(row[mutation_col])
            mutant_seq = None
            if mutant_col and mutant_col != mutation_col and pd.notna(row.get(mutant_col, None)):
                mutant_seq = str(row[mutant_col])
            fitness = (
                float(row[fitness_col])
                if (fitness_col and pd.notna(row.get(fitness_col, None)))
                else None
            )
            assay_id = (
                row[assay_col]
                if (assay_col and pd.notna(row.get(assay_col, None)))
                else None
            )

            # parse mutation string
            parsed = None
            if mut_str:
                try:
                    parsed = parse_mutation_string(mut_str)
                except Exception as exc:
                    logger.warning("Failed to parse mutation string %r: %s", mut_str, exc)

            # reconstruct mutant sequence from wt + mutations if not already present
            if mutant_seq is None and parsed is not None and wt_seq is not None:
                try:
                    mutant_seq = apply_mutations(wt_seq, parsed)
                except Exception as exc:
                    logger.warning("Failed to apply mutations for %r: %s", mut_str, exc)

            # infer mutation string from sequence diff if no explicit mutation string
            if parsed is None and wt_seq and mutant_seq and len(wt_seq) == len(mutant_seq):
                diffs = [
                    f"{a}{i + 1}{b}"
                    for i, (a, b) in enumerate(zip(wt_seq, mutant_seq))
                    if a != b
                ]
                if diffs:
                    try:
                        parsed = parse_mutation_string(":".join(diffs))
                    except Exception as exc:
                        logger.warning("Failed to infer mutation string from diff: %s", exc)

            # route fitness to the correct target key based on assay type
            assay_type = self.assay_type_map.get(
                str(assay_id) if assay_id is not None else "",
                AssayType.ORGANISMAL_FITNESS,
            )
            targets = _make_targets(fitness, assay_type)

            # encode action tensor
            action = None
            if parsed is not None:
                try:
                    action = encode_mutation(parsed)
                except Exception as exc:
                    logger.warning("Failed to encode mutation %r: %s", mut_str, exc)

            rec: Dict[str, Any] = {
                "wt_sequence":    wt_seq,
                "mutation_string": mut_str,
                "mutant_sequence": mutant_seq,  
                "parsed_mutation": parsed, 
                "targets":         targets,
                "assay_id":        assay_id,
                "action":          action,
            }

            # pre-tokenize if tokenizer is available and pretokenize=True
            if tokenizer is not None and pretokenize:
                rec = self._tokenize_record(rec)

            self._records.append(rec)

    def _tokenize_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """tokenize wt_sequence and mutant_sequence and store in the record"""
        from .preprocessing import tokenize_sequence
        try:
            if isinstance(self.tokenizer, str):
                rec["input_wt"] = tokenize_sequence(
                    rec["wt_sequence"],
                    model_name=self.tokenizer,
                    return_tensors=self.return_tensors,
                )
                if rec.get("mutation_string"):
                    from ..tokenizers import ProteinTokenizer
                    ptok = ProteinTokenizer({"mode": self.tokenizer_mode})
                    rec["input_mutant"] = ptok.encode_mutation_string(
                        rec["wt_sequence"],
                        rec["mutation_string"],
                        return_tensors=self.return_tensors,
                    )
                else:
                    rec["input_mutant"] = None
            else:
                tok = self.tokenizer
                rec["input_wt"] = tok.encode_sequence(
                    rec["wt_sequence"],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors=self.return_tensors,
                )
                if rec.get("mutation_string"):
                    rec["input_mutant"] = tok.encode_mutation_string(
                        rec["wt_sequence"],
                        rec["mutation_string"],
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors=self.return_tensors,
                        strict_wt_check=self.strict_wt_check,
                    )
                else:
                    rec["input_mutant"] = None
        except Exception as exc:
            logger.warning("Tokenization failed: %s", exc)
            rec.setdefault("input_wt", None)
            rec.setdefault("input_mutant", None)
        return rec

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = dict(self._records[idx])

        if self.load_structures:
            from .preprocessing import load_structure
            if rec.get("wt_sequence") and "wt_structure" not in rec:
                try:
                    rec["wt_structure"] = load_structure(
                        rec["wt_sequence"], cache_dir=self.struct_cache_dir
                    )
                except Exception as exc:
                    logger.warning("Could not load wt_structure for idx %d: %s", idx, exc)
                    rec["wt_structure"] = None
            if rec.get("mutant_sequence") and "mutant_structure" not in rec:
                try:
                    rec["mutant_structure"] = load_structure(
                        rec["mutant_sequence"], cache_dir=self.struct_cache_dir
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not load mutant_structure for idx %d: %s", idx, exc
                    )
                    rec["mutant_structure"] = None

        # tokenize on-the-fly only when pretokenize=False
        if self.tokenizer is not None and "input_wt" not in rec:
            rec = self._tokenize_record(rec)

        return rec



class TsuboyamaDataset(ProteinGymDataset):
    """loads Tsuboyama 2023 mega-scale stability data"""
    def __init__(self, config: dict, path: "str | pd.DataFrame", **kwargs) -> None:
        # force all rows to STABILITY regardless of any provided assay_type_map
        kwargs.setdefault("assay_type_map", {})
        super().__init__(config, path, **kwargs)
        for rec in self._records:
            all_vals = [v for v in rec["targets"].values() if v == v]  # non-nan values
            fitness = all_vals[0] if all_vals else float("nan")
            rec["targets"] = {
                "stability": fitness,
                "affinity":  float("nan"),
                "activity":  float("nan"),
                "fitness":   float("nan"),
            }



class FitnessTransitionDataset(Dataset):
    """constructs (s_t, action, s_t1, reward, targets) rl transition tuples"""
    def __init__(self, base_dataset: ProteinGymDataset) -> None:
        if not isinstance(base_dataset, ProteinGymDataset):
            raise TypeError(
                f"FitnessTransitionDataset expects a ProteinGymDataset, "
                f"got {type(base_dataset)}"
            )
        self.base = base_dataset
        self._transitions: List[Dict[str, Any]] = []

        for i in range(len(self.base)):
            s = self.base[i]
            if not (s.get("wt_sequence") and s.get("mutant_sequence")):
                continue
            targets = s.get("targets", {"stability": float("nan"),
                                         "affinity":  float("nan"),
                                         "activity":  float("nan")})
            # primary reward = first non-NaN target value
            reward = next(
                (v for v in targets.values() if v == v), float("nan")  # v==v is False for NaN
            )
            self._transitions.append({
                "s_t":     {"sequence": s["wt_sequence"]},
                "action":  s.get("action"),
                "s_t1":    {"sequence": s["mutant_sequence"]},
                "reward":  reward,
                "targets": targets,
                "assay_id": s.get("assay_id"),
            })

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._transitions[idx]


class CustomAssayDataset(ProteinGymDataset):
    """dataset for user-uploaded csv files with `sequence` and `fitness` columns"""
    def __init__(self, config: dict, path: "str | pd.DataFrame", **kwargs) -> None:
        kwargs.setdefault("seq_col", config.get("seq_col", "sequence"))
        kwargs.setdefault("fitness_col", config.get("fitness_col", "fitness"))
        super().__init__(config, path, **kwargs)
