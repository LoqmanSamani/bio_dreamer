"""
Wraps a protein fitness landscape as a gym-like environment for RL training
and evaluation. Provides the "real" environment that the world model learns to
approximate.

During world model training, the environment supplies ground-truth transitions
from DMS data or an oracle. During policy training, the agent operates entirely
inside the world model (dreaming) — the environment is NOT called. The
environment is used only for final evaluation and active-learning data collection.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from biodreamer.core.environment import BaseEnvironment, BaseOracle, BaseStructureOracle
from .config import ProteinDreamerConstants as _Constants

logger = logging.getLogger(__name__)

_AA_LIST: List[str] = _Constants.amino_acids()
_AA_TO_IDX: Dict[str, int] = _Constants.aa_to_idx()

State = Dict[str, Any]






class ProteinEnvironment(BaseEnvironment):
    """Gym-like environment wrapping a protein fitness landscape.

    models a protein design MDP where each action is a single-site amino acid
    substitution. the environment is the ground truth; the world model
    approximates it during dreaming.

    Episode lifecycle::

        env = ProteinEnvironment(wt, fitness_oracle)
        state = env.reset()
        for _ in range(n_steps):
            next_state, reward, done, info = env.step("A42G")
            if done:
                break

    State dict keys (compatible with ProteinEncoder.embed_observation()):
        ``sequence`` (str), ``fitness`` (float), ``coords`` (np.ndarray or None),
        ``plddt`` (np.ndarray or None), ``ptm`` (float or None).

    Args:
        wild_type_sequence: starting protein (unmodified baseline).
        fitness_oracle: maps sequence → scalar fitness.
        structure_oracle: optional structure predictor. When None the state
            coords/plddt/ptm fields will be None throughout the episode.
        max_steps: maximum mutations per episode.
        fitness_threshold: stop early when reward ≥ this value.
        track_trajectory: accumulate all (action, state, reward) tuples in
            self.trajectory for active-learning data collection.
    """
    def __init__(
        self,
        wild_type_sequence: str,
        fitness_oracle: BaseOracle,
        config: Any = None,
        structure_oracle: Optional[BaseStructureOracle] = None,
    ) -> None:
        self.max_steps         = config.get("max_steps", 50)
        self.fitness_threshold = config.get("fitness_threshold", None)
        self.track_trajectory  = config.get("track_trajectory", False)
        if not wild_type_sequence:
            raise ValueError("wild_type_sequence must be non-empty")
        if not isinstance(fitness_oracle, BaseOracle):
            raise TypeError(
                f"fitness_oracle must be a BaseOracle subclass, got {type(fitness_oracle)}"
            )
        self.wild_type_sequence = wild_type_sequence
        self.fitness_oracle = fitness_oracle
        self.structure_oracle = structure_oracle
        self._current_state: Optional[State] = None
        self._step_count: int = 0
        self._trajectory: List[Dict[str, Any]] = []

    def reset(self) -> State:
        """reset the episode to the wild-type sequence and return its state."""
        self._step_count = 0
        self._trajectory = []
        self._current_state = self._build_state(self.wild_type_sequence)
        return self._current_state

    def step(
        self,
        action: Union[str, Dict[str, Any]],
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """apply a mutation (single- or multi-site) and return the transition.

        args:
            action: mutation string ``"A42G"`` or multi-site ``"A42G:L100V"``
                (1-based, ProteinGym / Tsuboyama convention), OR a dict
                ``{position: int, aa_old: int, aa_new: int}`` (0-based).
                dict values may also be Pytorch scalars/tensors.

        returns:
            (next_state, reward, done, info)
            next_state: state dict for the mutant sequence.
            reward: fitness score from the oracle (float, possibly NaN).
            done: True when max_steps reached, threshold exceeded, or
                         invalid mutation.
            info: diagnostic dict with keys:
                         mutation_str, hamming, step, wt_fitness, n_mutations.
        """
        if self._current_state is None:
            raise RuntimeError("reset() must be called before step()")

        current_seq = self._current_state["sequence"]

        try:
            mutations, mutation_str = self._resolve_action(action, current_seq)
            next_seq = self._apply_mutations(current_seq, mutations)
        except Exception as exc:
            logger.warning("Invalid action %r: %s", action, exc)
            info: Dict[str, Any] = {
                "mutation_str": str(action),
                "error": str(exc),
                "step": self._step_count,
                "hamming": 0,
                "n_mutations": 0,
                "wt_fitness": self._current_state["fitness"],
            }
            return self._current_state, float("nan"), True, info

        next_state = self._build_state(next_seq)
        reward = next_state["fitness"]
        self._step_count += 1

        done = self._step_count >= self.max_steps
        if (
            self.fitness_threshold is not None
            and not np.isnan(reward)
            and reward >= self.fitness_threshold
        ):
            done = True

        info = {
            "mutation_str": mutation_str,
            "hamming": _hamming(current_seq, next_seq),
            "n_mutations": len(mutations),
            "step": self._step_count,
            "wt_fitness": self._current_state["fitness"],
        }

        if self.track_trajectory:
            action_stored = mutations[0] if len(mutations) == 1 else mutations
            self._trajectory.append(
                {
                    "action": action_stored,
                    "mutation_str": mutation_str,
                    "state": next_state,
                    "reward": reward,
                }
            )

        self._current_state = next_state
        return next_state, reward, done, info

    def render(self) -> Dict[str, Any]:
        """return sequence alignment and metadata for visualisation/logging.

        returns a plain dict::

            alignment: pairwise string "|" (match) / "X" (mismatch)
            sequence: current sequence
            wt_sequence: wild-type sequence
            mutations: list of mutation strings relative to wt ("A42G", …)
            fitness: current fitness score
            step: current step count
            hamming: Hamming distance from wt
        """
        if self._current_state is None:
            raise RuntimeError("reset() must be called before render()")

        current_seq = self._current_state["sequence"]
        wt = self.wild_type_sequence
        alignment = "".join("|" if a == b else "X" for a, b in zip(wt, current_seq))

        return {
            "alignment": alignment,
            "sequence": current_seq,
            "wt_sequence": wt,
            "mutations": _diff_to_mutation_list(wt, current_seq),
            "fitness": self._current_state["fitness"],
            "step": self._step_count,
            "hamming": _hamming(wt, current_seq),
        }


    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        """all (action, state, reward) records accumulated since last reset()"""
        return list(self._trajectory)

    @property
    def current_state(self) -> Optional[State]:
        return self._current_state


    def _build_state(self, sequence: str) -> State:
        fitness = self.fitness_oracle.query(sequence)
        coords: Optional[np.ndarray] = None
        plddt: Optional[np.ndarray] = None
        ptm: Optional[float] = None

        if self.structure_oracle is not None:
            try:
                coords, plddt, ptm = self.structure_oracle.predict(sequence)
            except Exception as exc:
                logger.warning(
                    "Structure oracle failed for sequence of length %d: %s",
                    len(sequence), exc,
                )

        return {
            "sequence": sequence,
            "fitness": fitness,
            "coords": coords,
            "plddt": plddt,
            "ptm": ptm,
        }

    def _resolve_action(
        self,
        action: Union[str, Dict[str, Any]],
        current_seq: str,
    ) -> Tuple[List[Dict[str, int]], str]:
        """normalise action → (list of {position, aa_old, aa_new}, mutation_str).

        string actions may be multi-site ("A42G:L100V"), all sites are returned.
        dict actions are single-site by definition.
        positions are 0-based; mutation_str uses 1-based ProteinGym convention.
        """
        if isinstance(action, str):
            from .data.preprocessing import parse_mutation_string
            parsed = parse_mutation_string(action)
            if not parsed.records:
                raise ValueError(f"No mutation records parsed from {action!r}")
            mutations = [
                {
                    "position": rec.position_0,
                    "aa_old": _AA_TO_IDX.get(rec.wt_aa.upper(), 0),
                    "aa_new": _AA_TO_IDX.get(rec.mut_aa.upper(), 0),
                }
                for rec in parsed.records
                if not rec.is_deletion  # skip indels in substitution-only env
            ]
            if not mutations:
                raise ValueError(f"No substitution records in {action!r} (only indels)")
            return mutations, action

        # dict action, single-site, values may be plain ints or PyTorch tensors
        pos = int(action["position"]) if not isinstance(action["position"], int) else action["position"]
        aa_old = _aa_int(action["aa_old"])
        aa_new = _aa_int(action["aa_new"])

        wt_aa = _AA_LIST[aa_old] if 0 <= aa_old < len(_AA_LIST) else "?"
        mut_aa = _AA_LIST[aa_new] if 0 <= aa_new < len(_AA_LIST) else "?"
        mut_str = f"{wt_aa}{pos + 1}{mut_aa}"
        return [{"position": pos, "aa_old": aa_old, "aa_new": aa_new}], mut_str

    def _apply_mutations(self, sequence: str, mutations: List[Dict[str, int]]) -> str:
        """apply a list of substitutions sequentially, raises on out-of-bounds"""
        seq = sequence
        for m in mutations:
            pos = m["position"]
            aa_new = m["aa_new"]
            if not (0 <= pos < len(seq)):
                raise IndexError(
                    f"position={pos} out of bounds for sequence length {len(seq)}"
                )
            new_aa = _AA_LIST[aa_new] if 0 <= aa_new < len(_AA_LIST) else "?"
            seq = seq[:pos] + new_aa + seq[pos + 1:]
        return seq







class CachedOracle(BaseOracle):
    """wraps any oracle with an in-memory lru-style cache keyed by sequence.

    essential for expensive oracles (ESMFold, predictor models) that are called
    repeatedly during evaluation or active learning.

    args:
        oracle: the underlying oracle to wrap.
    """

    def __init__(self, oracle: BaseOracle) -> None:
        self._oracle = oracle
        self._cache: Dict[str, float] = {}

    def query(self, sequence: str) -> float:
        if sequence not in self._cache:
            self._cache[sequence] = self._oracle.query(sequence)
        return self._cache[sequence]

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)






class DMSLookupOracle(BaseOracle):
    """exact fitness lookup from a DMS dataset.

    works with both ProteinGym and Tsuboyama data.  Returns float('nan') for
    variants absent from the dataset.

    The lookup supports two addressing modes:
      1. Direct key match: the query string is a mutant string like "A42G" or
         a colon-separated multi-site string "A42G:L100V".
      2. Sequence diff: when the query is a full-length sequence that differs
         from wt, the oracle computes the mutant string and tries a lookup.

    DataFrame column auto-detection covers both ProteinGym (``DMS_score``) and
    Tsuboyama (``ddG``, ``dG``, ``ddg``) naming conventions.

    args:
        dms_data: ``dict[mutant_str, fitness]`` or a ``pd.DataFrame`` with
            mutant-string and fitness columns.
        wt_sequence: wild-type amino acid sequence. Maps to 0.0 unless
            explicitly supplied in dms_data.
        normalize: z-score normalize fitness values at construction time.
        negate: negate all fitness values after loading.  Set ``True`` when
            using Tsuboyama ΔΔG data (where lower = more stable) and the RL
            reward convention is "higher = better".
    """

    _MUT_CANDIDATES = ["mutant", "mutation", "variant", "mut_str", "mut"]
    _SCORE_CANDIDATES = [
        "DMS_score", "fitness", "score", "delta_fitness", "DMS_score_bin",
        "ddG", "dG", "ddg", "delta_G", "dG_ML",
    ]
    def __init__(
        self,
        dms_data: Any,
        wt_sequence: str,
        normalize: bool = False,
        negate: bool = False,
    ) -> None:
        self.wt_sequence = wt_sequence
        self.negate = negate

        if isinstance(dms_data, dict):
            self._lookup: Dict[str, float] = {k: float(v) for k, v in dms_data.items()}
        else:
            try:
                mut_col = _find_df_col(dms_data, self._MUT_CANDIDATES)
                score_col = _find_df_col(dms_data, self._SCORE_CANDIDATES)
                if mut_col is None or score_col is None:
                    raise ValueError(
                        f"Cannot find mutant/score columns. "
                        f"Available: {list(dms_data.columns)}"
                    )
                self._lookup = {
                    str(k): float(v)
                    for k, v in zip(dms_data[mut_col], dms_data[score_col])
                }
            except (AttributeError, TypeError) as exc:
                raise ValueError(f"Cannot parse dms_data: {exc}") from exc

        if wt_sequence not in self._lookup:
            self._lookup[wt_sequence] = 0.0

        if negate:
            self._lookup = {k: -v if not np.isnan(v) else float("nan")
                            for k, v in self._lookup.items()}

        if normalize:
            vals = np.array(
                [v for v in self._lookup.values() if not np.isnan(v)], dtype=np.float64
            )
            if len(vals) > 1:
                mu, sigma = float(vals.mean()), float(vals.std())
                sigma = max(sigma, 1e-8)
                self._lookup = {
                    k: (v - mu) / sigma if not np.isnan(v) else float("nan")
                    for k, v in self._lookup.items()
                }

    def query(self, sequence: str) -> float:
        if sequence in self._lookup:
            return self._lookup[sequence]
        # attempt sequence-diff lookup
        if len(sequence) == len(self.wt_sequence):
            mut_str = _sequence_to_mutant_string(sequence, self.wt_sequence)
            if mut_str == "":
                return self._lookup.get(self.wt_sequence, 0.0)
            if mut_str in self._lookup:
                return self._lookup[mut_str]
        return float("nan")

    @property
    def n_variants(self) -> int:
        return len(self._lookup)





class TsuboyamaDMSOracle(DMSLookupOracle):
    """convenience oracle for Tsuboyama 2023 mega-scale stability data.

    Tsuboyama measures ΔΔG in kcal/mol, where negative values indicate
    stabilising mutations.  This oracle negates the raw values by default so
    that the RL reward convention ("higher = better") is satisfied: a negative
    ΔΔG mutation receives a positive reward.

    all targets route to ``targets["stability"]`` (matching ``TsuboyamaDataset``).

    args:
        dms_data: Tsuboyama CSV/DataFrame or ``dict[mutant_str, ddG]``.
        wt_sequence: wild-type sequence.
        negate: negate ΔΔG scores so that higher reward = more stable.
            Defaults to True.  Set False if the values have already been negated
            upstream (e.g. by the training data pipeline).
        normalize: z-score normalize after negation.
    """

    def __init__(
        self,
        dms_data: Any,
        wt_sequence: str,
        negate: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            dms_data,
            wt_sequence=wt_sequence,
            normalize=normalize,
            negate=negate,
        )





class ESMFoldOracle(BaseOracle):
    """uses ESMFold mean pLDDT as a designability / fitness proxy.

    higher mean pLDDT indicates a more confidently-folded structure, which
    correlates with thermodynamic stability and expression level.  Suitable when
    no experimental DMS data is available.

    args:
        cache_dir: directory for caching ESMFold predictions.
        device: torch device for ESMFold inference.
        baseline_plddt: wt mean pLDDT subtracted to yield Δ pLDDT scores.
            Call set_baseline(wt_sequence) to set this automatically.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Any = None,
        baseline_plddt: Optional[float] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.device = device
        self.baseline_plddt = baseline_plddt

    def query(self, sequence: str) -> float:
        from .data.preprocessing import predict_structure_esmfold
        try:
            _, plddt, _ = predict_structure_esmfold(
                sequence, cache_dir=self.cache_dir, device=self.device
            )
            if plddt is None:
                return float("nan")
            score = float(np.mean(plddt))
            if self.baseline_plddt is not None:
                score -= self.baseline_plddt
            return score
        except Exception as exc:
            logger.warning(
                "ESMFoldOracle.query failed for sequence of length %d: %s",
                len(sequence), exc,
            )
            return float("nan")

    def set_baseline(self, wt_sequence: str) -> float:
        """run ESMFold on wt and store its mean pLDDT as baseline.

        Subsequent query() calls return Δ pLDDT (score – baseline).
        Returns the baseline pLDDT value.
        """
        # temporarily clear baseline so we get absolute pLDDT
        self.baseline_plddt = None
        score = self.query(wt_sequence)
        if not np.isnan(score):
            self.baseline_plddt = score
        return score





class PredictorOracle(BaseOracle):
    """wraps a callable fitness predictor as an oracle.

    The predictor is called as ``predictor(sequence: str) -> float``.
    Examples: Tranception log-likelihood, ESM-2 zero-shot pseudo-perplexity,
    a fine-tuned regression head loaded outside the env.

    args:
        predictor: callable with signature ``(sequence: str) -> float``.
    """

    def __init__(self, predictor: Any) -> None:
        if not callable(predictor):
            raise TypeError(f"predictor must be callable, got {type(predictor)}")
        self._predictor = predictor

    def query(self, sequence: str) -> float:
        try:
            return float(self._predictor(sequence))
        except Exception as exc:
            logger.warning("PredictorOracle.query failed: %s", exc)
            return float("nan")





class WetLabOracle(BaseOracle):
    """placeholder oracle for experimental fitness measurements.

    in production this interfaces with a LIMS or lab automation API.
    Sequences without registered measurements return float('nan').

    usage::

        oracle = WetLabOracle()
        oracle.register_measurement("ACDE...", 1.23)
        fitness = oracle.query("ACDE...")
    """

    def __init__(self) -> None:
        self._measurements: Dict[str, float] = {}

    def register_measurement(self, sequence: str, fitness: float) -> None:
        self._measurements[sequence] = float(fitness)

    def register_batch(self, measurements: Dict[str, float]) -> None:
        for seq, fit in measurements.items():
            self._measurements[seq] = float(fit)

    def query(self, sequence: str) -> float:
        return self._measurements.get(sequence, float("nan"))

    @property
    def n_measured(self) -> int:
        return len(self._measurements)






class ESMFoldStructureOracle(BaseStructureOracle):
    """wraps predict_structure_esmfold for use as a structure oracle.

    args:
        coord_mode: "ca", "backbone", or "all_atom" (default "ca").
        cache_dir: directory for caching predictions.
        device: torch device for inference.
    """

    def __init__(
        self,
        coord_mode: str = "ca",
        cache_dir: Optional[str] = None,
        device: Any = None,
    ) -> None:
        self.coord_mode = coord_mode
        self.cache_dir = cache_dir
        self.device = device

    def predict(
        self, sequence: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
        from .data.preprocessing import predict_structure_esmfold
        return predict_structure_esmfold(
            sequence,
            coord_mode=self.coord_mode,  # type: ignore[arg-type]
            cache_dir=self.cache_dir,
            device=self.device,
        )






# helpers
def _hamming(seq1: str, seq2: str) -> int:
    return sum(a != b for a, b in zip(seq1, seq2))


def _find_df_col(df: Any, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sequence_to_mutant_string(mutant: str, wt: str) -> str:
    """colon-separated mutation string for all positions where mutant differs from wt.

    positions are 1-based (ProteinGym convention): "A42G:L100V".
    returns "" when the sequences are identical.
    """
    parts = [
        f"{wt_aa}{i + 1}{mut_aa}"
        for i, (wt_aa, mut_aa) in enumerate(zip(wt, mutant))
        if wt_aa != mut_aa
    ]
    return ":".join(parts)


def _diff_to_mutation_list(wt: str, mutant: str) -> List[str]:
    """list of 1-based mutation strings for positions where wt and mutant differ."""
    return [
        f"{wt_aa}{i + 1}{mut_aa}"
        for i, (wt_aa, mut_aa) in enumerate(zip(wt, mutant))
        if wt_aa != mut_aa
    ]


def _aa_int(val: Any) -> int:
    """coerce a value (int, long tensor, or str) to a 0-based aa index."""
    if isinstance(val, str):
        return _AA_TO_IDX.get(val.upper(), 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
