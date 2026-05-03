from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from ...core.tokenizers import BaseProteinTokenizer
    AA_LIST: List[str] = list(BaseProteinTokenizer.AMINO_ACIDS)
except Exception:
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

AA_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(AA_LIST)}

_BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O"})
_ATOM37_BACKBONE_IDX: List[int] = [0, 1, 2, 4]
_ATOM37_CA_IDX: int = 1

CoordMode = Literal["ca", "backbone", "all_atom"]





def tokenize_sequence(
    sequence: str,
    model_name: str = "esm2-650m",
    return_tensors: bool = False,
    padding: bool = True,
    truncation: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """tokenize a protein sequence via an esm-style tokenizer.
    falls back to char-level encoding if the model tokenizer is unavailable.
    """
    try:
        from ..model_loader import HFModelLoader
        loader = HFModelLoader(device=device or torch.device("cpu"))
        model, tok = loader.load(model_name)
        if tok is None:
            from transformers import AutoTokenizer
            cfg = HFModelLoader.REGISTRY.get(model_name)
            hf_id = cfg.hf_id if cfg is not None else model_name
            tok = AutoTokenizer.from_pretrained(hf_id)
        kwargs = {"return_tensors": "pt"} if return_tensors else {}
        return tok(sequence, padding=padding, truncation=truncation, **kwargs)
    except Exception as exc:
        logger.warning("Tokenizer unavailable for '%s', using char-level fallback: %s", model_name, exc)
        ids = [AA_TO_IDX.get(c.upper(), 0) for c in sequence]
        mask = [1] * len(ids)
        if return_tensors:
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}


def parse_mutation_string(mut_str: str):
    """delegate to the tokenizers module's canonical parser"""
    from ..tokenizers import parse_mutation_string as _parse
    return _parse(mut_str)


def apply_mutations(wt_sequence: str, parsed_mutation) -> str:
    """apply a ParsedMutation to a wild-type sequence and return the mutant"""
    from ..tokenizers import apply_mutations as _apply
    return _apply(wt_sequence, parsed_mutation)


def encode_mutation(mutation: Any) -> Dict[str, torch.Tensor]:
    """encode a mutation specification into tensors for model input"""
    from ..tokenizers import MutationRecord, ParsedMutation

    if isinstance(mutation, str):
        parsed = parse_mutation_string(mutation)
        if not parsed.records:
            raise ValueError(f"No mutation records parsed from string: {mutation!r}")
        rec = parsed.records[0]
    elif isinstance(mutation, ParsedMutation):
        rec = mutation.records[0]
    elif isinstance(mutation, MutationRecord):
        rec = mutation
    else:
        raise TypeError(
            f"mutation must be str, MutationRecord, or ParsedMutation, got {type(mutation)}"
        )

    return {
        "position": torch.tensor(rec.position_0, dtype=torch.long),
        "aa_old":   torch.tensor(_aa_to_index(rec.wt_aa),  dtype=torch.long),
        "aa_new":   torch.tensor(_aa_to_index(rec.mut_aa), dtype=torch.long),
    }


def _aa_to_index(aa: str) -> int:
    return AA_TO_IDX.get((aa or "").upper(), 0)


def _ensure_cache_dir(cache_dir: Optional[str]) -> Path:
    if cache_dir is None:
        cache_dir = os.environ.get(
            "BIODREAMER_STRUCT_CACHE", "~/.cache/biodreamer/structures"
        )
    p = Path(cache_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def _select_coords_atom37(
    all_atom_positions: np.ndarray,
    coord_mode: CoordMode,
) -> np.ndarray:
    """extract coordinates from an ESMFold atom37 array of shape (L, 37, 3)"""
    arr = np.asarray(all_atom_positions, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 37 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected atom37 array of shape (L, 37, 3), got {arr.shape}"
        )
    if coord_mode == "ca":
        return arr[:, _ATOM37_CA_IDX, :]
    if coord_mode == "backbone":
        return arr[:, _ATOM37_BACKBONE_IDX, :]
    return arr


def _select_coords_pdb(
    atom_records: List[Dict],
    coord_mode: CoordMode,
) -> np.ndarray:
    """build a coordinate array from parsed pdb atom records"""
    if not atom_records:
        raise ValueError("No atom records provided")

    res_atoms: Dict[int, List[Dict]] = defaultdict(list)
    res_order: List[int] = []
    for rec in atom_records:
        rseq = rec["res_seq"]
        if rseq not in res_atoms:
            res_order.append(rseq)
        res_atoms[rseq].append(rec)

    L = len(res_order)

    if coord_mode == "ca":
        coords = np.full((L, 3), np.nan, dtype=np.float32)
        for i, rseq in enumerate(res_order):
            for rec in res_atoms[rseq]:
                if rec["atom_name"] == "CA":
                    coords[i] = [rec["x"], rec["y"], rec["z"]]
                    break
        if np.all(np.isnan(coords)):
            raise RuntimeError("No C-alpha atoms found in PDB records")
        return coords

    if coord_mode == "backbone":
        bb_order = ["N", "CA", "C", "O"]
        coords = np.full((L, 4, 3), np.nan, dtype=np.float32)
        for i, rseq in enumerate(res_order):
            atom_map = {r["atom_name"]: r for r in res_atoms[rseq]}
            for j, name in enumerate(bb_order):
                if name in atom_map:
                    r = atom_map[name]
                    coords[i, j] = [r["x"], r["y"], r["z"]]
        return coords

    out = np.empty(L, dtype=object)
    for i, rseq in enumerate(res_order):
        recs = res_atoms[rseq]
        out[i] = np.array([[r["x"], r["y"], r["z"]] for r in recs], dtype=np.float32)
    return out


def load_structure(
    pdb_path_or_sequence: str,
    coord_mode: CoordMode = "backbone",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """load protein coordinates from a PDB file or by ESMFold prediction"""
    p = Path(pdb_path_or_sequence)
    if p.exists():
        atom_records: List[Dict] = []
        with open(p) as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom_name = line[12:16].strip()
                if atom_name.startswith("H") or atom_name.startswith("D"):
                    continue
                try:
                    res_seq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                atom_records.append(
                    {"atom_name": atom_name, "res_seq": res_seq, "x": x, "y": y, "z": z}
                )
        if not atom_records:
            raise RuntimeError(f"No heavy atoms found in {p}")
        return _select_coords_pdb(atom_records, coord_mode)

    coords, _ = predict_structure_esmfold(
        pdb_path_or_sequence, coord_mode=coord_mode, device=device
    )
    return coords


def predict_structure_esmfold(
    sequence: str,
    coord_mode: CoordMode = "backbone",
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """predict protein structure with ESMFold, results are cached by sequence hash"""
    cache = _ensure_cache_dir(cache_dir)
    cache_file = cache / f"{_seq_hash(sequence)}.npz"

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        raw = data["coords"]
        plddt = data["plddt"] if "plddt" in data else None
        return _select_coords_atom37(raw, coord_mode), plddt

    tried: List[str] = []

    # esm package
    try:
        import esm
        try:
            model, _ = esm.pretrained.esmfold_v1()
            out = model.predict_structure(sequence)
            raw = np.asarray(out["coords"]).astype(np.float32)
            if raw.ndim == 2:
                raw = raw.reshape(len(sequence), -1, 3)
            plddt = (
                np.asarray(out["plddt"], dtype=np.float32)
                if out.get("plddt") is not None else None
            )
            np.savez_compressed(cache_file, coords=raw, plddt=plddt)
            return _select_coords_atom37(raw, coord_mode), plddt
        except Exception as e:
            tried.append(f"esm.pretrained call failed: {e}")
    except ImportError:
        tried.append("esm package not available")

    # HuggingFace ESMFold wrapper
    try:
        from ..model_loader import HFModelLoader
        loader = HFModelLoader(device=device or torch.device("cpu"))
        model, _ = loader.load("esmfold")
        for meth in ("predict_structure", "predict", "fold", "__call__", "forward"):
            try:
                fn = getattr(model, meth)
                out = fn(sequence)
                raw: Optional[np.ndarray] = None
                if isinstance(out, dict):
                    if "final_atom_positions" in out:
                        raw = np.asarray(out["final_atom_positions"]).astype(np.float32)
                    elif "coords" in out:
                        raw = np.asarray(out["coords"]).astype(np.float32)
                        if raw.ndim == 2:
                            raw = raw.reshape(len(sequence), -1, 3)
                    if raw is not None:
                        plddt = (
                            np.asarray(out["plddt"], dtype=np.float32)
                            if out.get("plddt") is not None else None
                        )
                        np.savez_compressed(cache_file, coords=raw, plddt=plddt)
                        return _select_coords_atom37(raw, coord_mode), plddt
                if hasattr(out, "detach") or isinstance(out, np.ndarray):
                    arr = (
                        out.detach().cpu().numpy()
                        if hasattr(out, "detach") else np.asarray(out)
                    )
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        raw = arr.astype(np.float32)
                        if raw.shape[1] != 37:
                            logger.warning(
                                "ESMFold fallback: unexpected atom axis size %d (expected 37)",
                                raw.shape[1],
                            )
                        np.savez_compressed(cache_file, coords=raw, plddt=None)
                        return _select_coords_atom37(raw, coord_mode), None
            except Exception:
                continue
        tried.append("HF ESMFold call methods exhausted")
    except Exception as e:
        tried.append(f"HFModelLoader/esmfold unavailable: {e}")

    raise RuntimeError(
        "Could not run ESMFold. Tried: " + "; ".join(tried) + ". "
        "Install the `esm` package or the HuggingFace ESMFold wrapper."
    )


def compute_distance_map(coords: np.ndarray) -> np.ndarray:
    """compute pairwise Euclidean distance map from a (L, 3) coordinate array"""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be shape (L, 3)")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def build_protein_graph(coords: np.ndarray, cutoff: float = 10.0):
    """build a residue-level contact graph from (L, 3) Cα coordinates"""
    coords = np.asarray(coords, dtype=np.float32)
    dmap = compute_distance_map(coords)
    rows, cols = np.where((dmap <= cutoff) & (dmap > 0.0))
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    edge_attr = dmap[rows, cols].astype(np.float32)
    x = torch.tensor(coords, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    try:
        from torch_geometric.data import Data
        return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    except ImportError:
        return {"x": x, "edge_index": edge_index_t, "edge_attr": edge_attr_t}


def normalize_fitness(raw_scores: List[float], method: str = "quantile") -> np.ndarray:
    """normalize fitness scores into [0, 1] (minmax / zscore / quantile)"""
    arr = np.asarray(raw_scores, dtype=float)
    if method == "minmax":
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        denom = mx - mn if mx != mn else 1.0
        return (arr - mn) / denom
    if method == "zscore":
        mu = np.nanmean(arr)
        sd = np.nanstd(arr) or 1.0
        return (arr - mu) / sd
    # quantile (rank-based) → [0, 1]
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    if len(arr) > 1:
        return ranks.astype(float) / (len(arr) - 1)
    return np.zeros_like(arr, dtype=float)


def augment_dms_data(dataset: Any, strategy: str = "none") -> Any:
    """data augmentation for dms datasets"""
    if strategy == "none":
        return dataset
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — cannot augment dataset")
        return dataset
    if not isinstance(dataset, pd.DataFrame):
        logger.warning("augment_dms_data: dataset is not a DataFrame — skipping")
        return dataset
    if strategy == "reverse":
        if {"wt_sequence", "mutant_sequence"}.issubset(dataset.columns):
            dup = dataset.copy().rename(
                columns={"wt_sequence": "mutant_sequence", "mutant_sequence": "wt_sequence"}
            )
            return pd.concat([dataset, dup], ignore_index=True)
        logger.warning("reverse augmentation requires 'wt_sequence' and 'mutant_sequence' columns")
        return dataset
    logger.warning("Unknown augmentation strategy '%s' — returning original dataset", strategy)
    return dataset
