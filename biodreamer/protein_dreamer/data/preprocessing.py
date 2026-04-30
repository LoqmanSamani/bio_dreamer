from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


try:
  from ..core.tokenizers import BaseProteinTokenizer
  AA_LIST = BaseProteinTokenizer.AMINO_ACIDS
except Exception:
  AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
_BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O"})
_ATOM37_BACKBONE_IDX: List[int] = [0, 1, 2, 4]
_ATOM37_CA_IDX: int = 1

CoordMode = Literal["ca", "backbone", "all_atom"]



def _ensure_cache_dir(cache_dir: Optional[str] | None) -> Path:
  """ensure the cache directory exists and return its path"""
  if cache_dir is None:
    cache_dir = os.environ.get("BIODREAMER_STRUCT_CACHE", "~/.cache/biodreamer/structures")
  p = Path(cache_dir).expanduser()
  p.mkdir(parents=True, exist_ok=True)
  return p


def _seq_hash(seq: str) -> str:
  """hash a sequence string into a fixed-length identifier for caching/persistence"""
  return hashlib.md5(seq.encode("utf-8")).hexdigest()


def _aa_to_index(aa: str) -> int:
  """get the integer index of an amino acid character, or 0 for unknown/ambiguous"""
  aa = (aa or "").upper()
  return AA_TO_IDX.get(aa, 0)


def tokenise_sequence(
  sequence: str,
  model_name: str = "esm2-650m",
  return_tensors: bool = False,
  padding: bool = True,
  truncation: bool = True,
  device: Optional[torch.device] = None,
) -> Dict[str, Any]:
  """tokenise a single protein sequence using an esm-style tokenizer.
  tries to load a tokenizer (`data.hfloader.HFModelLoader`) maching `model_name`. 
  false back to a simple char-level endocing if the tokenizer is unavailable.
  """
  try:
    from .hfloader import HFModelLoader
    loader = HFModelLoader(device=device or torch.device("cpu"))
    model, tok = loader.load(model_name)
    if tok is None:
      from transformers import AutoTokenizer
      cfg = HFModelLoader.REGISTRY.get(model_name)
      hf_id = cfg.hf_id if cfg is not None else model_name
      tok = AutoTokenizer.from_pretrained(hf_id)
    kwargs = {"return_tensors": "pt"} if return_tensors else {}
    return tok(sequence, padding=padding, truncation=truncation, **kwargs)
  except Exception:
    ids = [AA_TO_IDX.get(c.upper(), 0) for c in sequence]
    attention_mask = [1] * len(ids)
    if return_tensors:
      return {"input_ids": torch.tensor([ids], dtype=torch.long),
          "attention_mask": torch.tensor([attention_mask], dtype=torch.long)}
    return {"input_ids": ids, "attention_mask": attention_mask}


def parse_mutation_string(mut_str: str):
  """parse a mutation string like "A42G" into a structured format"""
  from .tokenizers.mutation_tokenizer import parse_mutation_string as _pms
  return _pms(mut_str)


def encode_mutation(mutation: Any) -> Dict[str, torch.Tensor]:
  """encode (tokenize) a mutation specification (string or structured) into tensors for model input"""
  from .tokenizers.mutation_tokenizer import MutationRecord, ParsedMutation
  if isinstance(mutation, str):
    parsed = parse_mutation_string(mutation)
    if not parsed.records:
      raise ValueError("No mutation records parsed from string")
    rec = parsed.records[0]
  elif isinstance(mutation, ParsedMutation):
    rec = mutation.records[0]
  elif isinstance(mutation, MutationRecord):
    rec = mutation
  else:
    raise TypeError("mutation must be str, MutationRecord, or ParsedMutation")
  # positional embedding layers expect float input (nn.Linear), embeddings expect long
  pos = torch.tensor(float(rec.position_0), dtype=torch.float)
  aa_old = torch.tensor(_aa_to_index(rec.wt_aa), dtype=torch.long)
  aa_new = torch.tensor(_aa_to_index(rec.mut_aa), dtype=torch.long)
  return {"position": pos, "aa_old": aa_old, "aa_new": aa_new}


def _select_coords_atom37(
  all_atom_positions: np.ndarray,
  coord_mode: CoordMode,
) -> np.ndarray:
  """select coordinates from an ESMFold atom37 array shaped (L, 37, 3).
  atom37 encodes up to 37 heavy-atom slots per residue in a fixed canonical
  order (N=0, CA=1, C=2, CB=3, O=4, …).  slots that are absent for a given
  residue are zero-filled by the model.
  returns
  -------
  - ``"ca"``       → (l, 3)
  - ``"backbone"`` → (l, 4, 3)  atoms in order n, ca, c, o
  - ``"all_atom"`` → (l, 37, 3)
  """
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
  """build a coordinate array from a list of parsed pdb-atom records.
  each record is a dict with keys `atom_name`, `res_seq` (int),
  `x`, `y`, `z`
  """
  if not atom_records:
    raise ValueError("No atom records provided")
  
  from collections import defaultdict
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
      atom_map = {rec["atom_name"]: rec for rec in res_atoms[rseq]}
      for j, name in enumerate(bb_order):
        if name in atom_map:
          rec = atom_map[name]
          coords[i, j] = [rec["x"], rec["y"], rec["z"]]
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
  """load protein coordinates from a pdb file or by predicting with esm-fold"""
  p = Path(pdb_path_or_sequence)
  if p.exists():
    atom_records: List[Dict] = []
    with open(p, "r") as fh:
      for line in fh:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
          continue
        atom_name = line[12:16].strip()
        # skip hydrogen atoms
        if atom_name.startswith("H") or atom_name.startswith("D"):
          continue
        try:
          res_seq = int(line[22:26])
          x = float(line[30:38])
          y = float(line[38:46])
          z = float(line[46:54])
        except Exception:
          continue
        atom_records.append({"atom_name": atom_name, "res_seq": res_seq,
                  "x": x, "y": y, "z": z})
    if not atom_records:
      raise RuntimeError(f"No heavy atoms found in {p}")
    return _select_coords_pdb(atom_records, coord_mode)
  # treat as sequence
  coords, plddt = predict_structure_esmfold(
    pdb_path_or_sequence, coord_mode=coord_mode, device=device
  )
  return coords


def predict_structure_esmfold(
  sequence: str,
  coord_mode: CoordMode = "backbone",
  cache_dir: Optional[str] = None,
  device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
  """predict structure using esm-fold and return coordinates in the requested mode"""
  cache = _ensure_cache_dir(cache_dir)
  h = _seq_hash(sequence)
  cache_file = cache / f"{h}.npz"
  if cache_file.exists():
    data = np.load(cache_file, allow_pickle=True)
    raw = data["coords"] # (l, 37, 3)
    plddt = data["plddt"] if "plddt" in data else None
    return _select_coords_atom37(raw, coord_mode), plddt

  tried: List[str] = []
  try:
    # esm package
    import esm
    try:
      model, alphabet = esm.pretrained.esmfold_v1()
      out = model.predict_structure(sequence)
      raw = np.asarray(out["coords"]).astype(np.float32) # (l, 37, 3)
      if raw.ndim == 2:
        L = len(sequence)
        raw = raw.reshape(L, -1, 3)
      plddt = (
        np.asarray(out["plddt"], dtype=np.float32)
        if out.get("plddt") is not None else None
      )
      np.savez_compressed(cache_file, coords=raw, plddt=plddt)
      return _select_coords_atom37(raw, coord_mode), plddt
    except Exception as e:
      tried.append(f"esm.pretrained call failed: {e}")
  except Exception:
    tried.append("esm package not available")
  # hugging-face wrapper
  try:
    from .hfloader import HFModelLoader
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
          arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
          if arr.ndim == 3 and arr.shape[2] == 3:
            raw = arr.astype(np.float32)
            if raw.shape[1] != 37:
              log.warning(
                "ESMFold fallback: unexpected atom axis size %d (expected 37); "
                "coord_mode='all_atom' will return raw output.",
                raw.shape[1],
              )
            plddt = None
            np.savez_compressed(cache_file, coords=raw, plddt=plddt)
            return _select_coords_atom37(raw, coord_mode), plddt
      except Exception:
        continue
    tried.append("hf esmfold call methods exhausted")
  except Exception as e:
    tried.append(f"hfloader/esmfold unavailable: {e}")

  raise RuntimeError(
    "Could not run ESMFold. Tried: " + "; ".join(tried)
    + ". Install the `esm` package or the HuggingFace ESMFold wrapper to enable "
    "structure prediction."
  )


def compute_distance_map(coords: np.ndarray) -> np.ndarray:
  """compute pairwise euclidean distance map from a `(l, 3)` coordinate array"""
  coords = np.asarray(coords, dtype=np.float32)
  if coords.ndim != 2 or coords.shape[1] != 3:
    raise ValueError("coords must be shape (L,3)")
  dif = coords[:, None, :] - coords[None, :, :]
  dmap = np.sqrt((dif ** 2).sum(-1))
  return dmap


def build_protein_graph(coords: np.ndarray, cutoff: float = 10.0):
  """build a residue-level contact graph from a `(l, 3)` coordinate array"""
  import torch
  coords = np.asarray(coords, dtype=np.float32)
  dmap = compute_distance_map(coords)
  L = coords.shape[0]
  rows, cols = np.where((dmap <= cutoff) & (dmap > 0.0))
  # create bidirectional edges
  edge_index = np.vstack([rows, cols]).astype(np.int64)
  edge_attr = dmap[rows, cols].astype(np.float32)
  x = torch.tensor(coords, dtype=torch.float32)
  edge_index_t = torch.tensor(edge_index, dtype=torch.long)
  edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
  try:
    from torch_geometric.data import Data
    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
  except Exception:
    return {"x": x, "edge_index": edge_index_t, "edge_attr": edge_attr_t}


def normalise_fitness(raw_scores: List[float], method: str = "quantile") -> np.ndarray:
  """normalise a list/array of fitness scores into the [0, 1] range using the specified method"""
  arr = np.asarray(raw_scores, dtype=float)
  if method == "minmax":
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    denom = mx - mn if mx != mn else 1.0
    return (arr - mn) / denom
  if method == "zscore":
    mu = np.nanmean(arr)
    sd = np.nanstd(arr) if np.nanstd(arr) != 0 else 1.0
    return (arr - mu) / sd
  # quantile (rank-based) mapping to [0,1]
  order = np.argsort(arr)
  ranks = np.empty_like(order)
  ranks[order] = np.arange(len(arr))
  if len(arr) > 1:
    return ranks.astype(float) / (len(arr) - 1)
  return np.zeros_like(arr, dtype=float)


def augment_dms_data(dataset: Any, strategy: str = "none") -> Any:
  """augmentation of dms datasets"""
  if strategy == "none":
    return dataset
  try:
    import pandas as pd
  except Exception:
    log.warning("pandas not available — cannot augment dataset")
    return dataset
  if not isinstance(dataset, pd.DataFrame):
    log.warning("augment_dms_data: dataset is not a pandas DataFrame — skipping augmentation")
    return dataset
  if strategy == "reverse":
    if {"wt_sequence", "mutant_sequence"}.issubset(set(dataset.columns)):
      dup = dataset.copy()
      dup = dup.rename(columns={"wt_sequence": "mutant_sequence", "mutant_sequence": "wt_sequence"})
      out = pd.concat([dataset, dup], ignore_index=True)
      return out
    else:
      log.warning("reverse augmentation requires 'wt_sequence' and 'mutant_sequence' columns")
      return dataset
  log.warning("Unknown augmentation strategy '%s' — returning original dataset", strategy)
  return dataset