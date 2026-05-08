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
    config: Any = None,
    model_name: Optional[str] = None,
    return_tensors: Optional[bool] = None,
    padding: Optional[bool] = None,
    truncation: Optional[bool] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """tokenize a protein sequence via an esm-style tokenizer.
    falls back to char-level encoding if the model tokenizer is unavailable.
    """
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _model_name     = model_name     if model_name     is not None else config.get("tokenizer_model", "esm2-150m")
    _return_tensors = return_tensors if return_tensors is not None else config.get("return_tensors",  False)
    _padding        = padding        if padding        is not None else config.get("padding",          True)
    _truncation     = truncation     if truncation     is not None else config.get("truncation",       True)
    try:
        from ..model_loader import HFModelLoader
        loader = HFModelLoader(device=device or torch.device("cpu"))
        model, tok = loader.load(_model_name)
        if tok is None:
            from transformers import AutoTokenizer
            cfg = HFModelLoader.REGISTRY.get(_model_name)
            hf_id = cfg.hf_id if cfg is not None else _model_name
            tok = AutoTokenizer.from_pretrained(hf_id)
        kwargs = {"return_tensors": "pt"} if _return_tensors else {}
        return tok(sequence, padding=_padding, truncation=_truncation, **kwargs)
    except Exception as exc:
        logger.warning("Tokenizer unavailable for '%s', using char-level fallback: %s", _model_name, exc)
        ids = [AA_TO_IDX.get(c.upper(), 0) for c in sequence]
        mask = [1] * len(ids)
        if _return_tensors:
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
    config: Any = None,
    coord_mode: Optional[CoordMode] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    """load protein coords from a PDB file or ESMFold"""
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _coord_mode = coord_mode if coord_mode is not None else config.get("coord_mode", "ca")
    _cache_dir  = cache_dir  if cache_dir  is not None else config.get("struct_cache_dir", None)
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
        return _select_coords_pdb(atom_records, _coord_mode), None, None

    return predict_structure_esmfold(
        pdb_path_or_sequence, coord_mode=_coord_mode, cache_dir=_cache_dir, device=device
    )


def _extract_ptm(out: dict) -> Optional[float]:
    """Safely extract the global pTM score from an ESMFold output dict."""
    for key in ("ptm", "predicted_tm_score", "tm_score"):
        val = out.get(key)
        if val is not None:
            try:
                return float(val.item() if hasattr(val, "item") else val)
            except Exception:
                pass
    return None


def _cache_save(cache_file, raw: np.ndarray, plddt: Optional[np.ndarray], ptm: Optional[float]) -> None:
    ptm_arr = np.array([ptm if ptm is not None else float("nan")], dtype=np.float32)
    np.savez_compressed(cache_file, coords=raw, plddt=plddt, ptm=ptm_arr)


def predict_structure_esmfold(
    sequence: str,
    config: Any = None,
    coord_mode: Optional[CoordMode] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    """predict protein structure with ESMFold; results are cached by sequence hash.

    returns `(coords, plddt, ptm)` where:
    - coords: Cα or backbone coordinates per coord_mode
    - plddt: per-residue confidence scores (0–100), or None
    - ptm: global predicted TM-score (0–1), or None
    """
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _coord_mode = coord_mode if coord_mode is not None else config.get("coord_mode", "ca")
    _cache_dir  = cache_dir  if cache_dir  is not None else config.get("struct_cache_dir", None)
    cache = _ensure_cache_dir(_cache_dir)
    cache_file = cache / f"{_seq_hash(sequence)}.npz"

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        raw = data["coords"]
        plddt = data["plddt"] if "plddt" in data else None
        ptm = None
        if "ptm" in data:
            ptm_arr = data["ptm"]
            v = float(ptm_arr.flat[0])
            ptm = v if not np.isnan(v) else None
        return _select_coords_atom37(raw, _coord_mode), plddt, ptm

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
            ptm = _extract_ptm(out)
            _cache_save(cache_file, raw, plddt, ptm)
            return _select_coords_atom37(raw, _coord_mode), plddt, ptm
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
                        ptm = _extract_ptm(out)
                        _cache_save(cache_file, raw, plddt, ptm)
                        return _select_coords_atom37(raw, _coord_mode), plddt, ptm
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
                        _cache_save(cache_file, raw, None, None)
                        return _select_coords_atom37(raw, _coord_mode), None, None
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


def build_protein_graph(
    coords: np.ndarray,
    config: Any = None,
    cutoff: Optional[float] = None,
    plddt: Optional[np.ndarray] = None,
):
    """build a residue-level contact graph from (L, 3) Cα coordinates.

    returns a dict (or torch_geometric.Data) with the following fields for
    GVP-GNN consumption:
    - node_s  (L, 1): pLDDT normalised to [0,1], or 1.0 if unavailable
    - node_v  (L, 1, 3): Cα position as an equivariant vector feature
    - edge_index (2, E)
    - edge_s  (E, 1): inter-residue distance (Å)
    - edge_v  (E, 1, 3): unit displacement vector from source to target

    backward-compatible fields:
    - x  (L, 3): raw Cα coordinates
    - edge_attr (E, 1): same as edge_s (distances)
    """
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _cutoff = cutoff if cutoff is not None else config.get("contact_cutoff", 10.0)
    coords = np.asarray(coords, dtype=np.float32)
    dmap = compute_distance_map(coords)
    rows, cols = np.where((dmap <= _cutoff) & (dmap > 0.0))

    # edge features
    distances = dmap[rows, cols].astype(np.float32)
    edge_s = torch.tensor(distances[:, None], dtype=torch.float32)  # (E, 1)
    disp = coords[cols] - coords[rows]  # (E, 3)
    norms = np.linalg.norm(disp, axis=1, keepdims=True).clip(min=1e-8)
    edge_v = torch.tensor((disp / norms)[:, None, :], dtype=torch.float32)  # (E, 1, 3)
    edge_index = torch.tensor(np.vstack([rows, cols]).astype(np.int64), dtype=torch.long)

    # node features
    if plddt is not None:
        node_s = torch.tensor(
            (np.asarray(plddt, dtype=np.float32) / 100.0)[:, None], dtype=torch.float32
        )  # (L, 1), division by 100 to normalise pLDDT to [0, 1]
    else:
        node_s = torch.ones(len(coords), 1, dtype=torch.float32)
    node_v = torch.tensor(coords[:, None, :], dtype=torch.float32)  # (L, 1, 3)

    # backward-compat
    x = torch.tensor(coords, dtype=torch.float32)

    try:
        from torch_geometric.data import Data
        return Data(
            x=x, node_s=node_s, node_v=node_v,
            edge_index=edge_index, edge_s=edge_s, edge_v=edge_v,
            edge_attr=edge_s,
        )
    except ImportError:
        return {
            "x": x, "node_s": node_s, "node_v": node_v,
            "edge_index": edge_index, "edge_s": edge_s, "edge_v": edge_v,
            "edge_attr": edge_s,
        }


def normalize_fitness(
    raw_scores: List[float],
    config: Any = None,
    method: Optional[str] = None,
) -> np.ndarray:
    """normalize fitness scores into [0, 1] (minmax / zscore / quantile)"""
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _method = method if method is not None else config.get("fitness_norm", "quantile")
    arr = np.asarray(raw_scores, dtype=float)
    if _method == "minmax":
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        denom = mx - mn if mx != mn else 1.0
        return (arr - mn) / denom
    if _method == "zscore":
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


def augment_dms_data(
    dataset: Any,
    config: Any = None,
    strategy: Optional[str] = None,
) -> Any:
    """data augmentation for dms datasets"""
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["preprocessing"]
    _strategy = strategy if strategy is not None else config.get("augmentation", "none")
    if _strategy == "none":
        return dataset
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — cannot augment dataset")
        return dataset
    if not isinstance(dataset, pd.DataFrame):
        logger.warning("augment_dms_data: dataset is not a DataFrame — skipping")
        return dataset
    if _strategy == "reverse":
        if {"wt_sequence", "mutant_sequence"}.issubset(dataset.columns):
            dup = dataset.copy().rename(
                columns={"wt_sequence": "mutant_sequence", "mutant_sequence": "wt_sequence"}
            )
            return pd.concat([dataset, dup], ignore_index=True)
        logger.warning("reverse augmentation requires 'wt_sequence' and 'mutant_sequence' columns")
        return dataset
    logger.warning("Unknown augmentation strategy '%s' — returning original dataset", _strategy)
    return dataset
