from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModel, AutoTokenizer,
    EsmModel, EsmTokenizer, EsmForProteinFolding,
    EsmForMaskedLM,               
    T5EncoderModel, T5Tokenizer,    
)
import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path
import requests
import yaml
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
log = logging.getLogger(__name__)
import os
try:
    import pandas as pd
except Exception:  
    pd = None
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  
    torch = None  
    Dataset = object
from .preprocessing import (
    tokenise_sequence,
    parse_mutation_string,
    encode_mutation,
    compute_distance_map,
    build_protein_graph,
    load_structure,
    augment_dms_data,
)

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Callable
import numpy as np
import math


try:
    import torch
    from torch.utils.data import DataLoader
except Exception: 
    torch = None 
    DataLoader = object


# boto3 for S3 downloads (falls back to HTTPS if not installed)
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config as BotocoreConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
    
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




# basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("protein_dreamer.download") # we may change it to "protein_dreamer.data.download" later if we move this file


# s3 keys for ProteinGym subsets
# maps config subset names → (S3 key, local filename)
PROTEINGYM_S3_KEYS: dict[str, tuple[str, str]] = {
    "DMS_substitutions":      ("DMS_substitutions.parquet",  "DMS_substitutions.parquet"),
    "DMS_indels":             ("DMS_indels.parquet",          "DMS_indels.parquet"),
    "clinical_substitutions": ("clinical_substitutions.parquet", "clinical_substitutions.parquet"),
    "clinical_indels":        ("clinical_indels.parquet",     "clinical_indels.parquet"),
}

# reference metadata files stored separately in the official github release
# (also mirrored on s3 – we fall back to github if s3 path missing)
PROTEINGYM_REF_URLS: dict[str, str] = {
    "ProteinGym_reference_file_substitutions.csv": (
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
        "reference_files/DMS_substitutions/ProteinGym_reference_file_substitutions.csv"
    ),
    "ProteinGym_reference_file_indels.csv": (
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
        "reference_files/DMS_indels/ProteinGym_reference_file_indels.csv"
    ),
}

# zenodo file names → download slugs
TSUBOYAMA_FILES: dict[str, str] = {
    "processed_datasets": "Processed_K50_dG_datasets.zip",
    "dataset1_csv":       "Tsuboyama2023_Dataset1_20230416.csv",
    "dataset2_3_csv":     "Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
    "single_dms_list":    "Single_DMS_list.csv",
    "double_dms_list":    "Double_DMS_list.csv",
    "triple_dms_list":    "Triple_DMS_list.csv",
}





class ModelTask(Enum):
    """enumeration of possible tasks for loaded models"""
    SEQUENCE_EMBEDDING = 'sequence_embedding'   # e.g., ESM-2, ProtTrans
    STRUCTURE_EMBEDDING = 'structure_embedding' # e.g., GVP, SaPort, GVP-GNN
    STRUCTURE_PREDICTION = 'structure_prediction' # e.g., ESMFold
    INVERSE_FOLDING = 'inverse_folding' # e.g., ESM-IF1, ESM-IF2
    FUNCTION_PREDICTION = 'function_prediction' # e.g., DeepFRI, ESM-2 function head
    OTHER = 'other' # for any other tasks not covered above
    
    
    
class ModelBackend(Enum):
    """enumeration of possible backends for loading models"""
    AUTO = 'auto'           # AutoModel + AutoTokenizer (generic fallback)
    ESM = 'esm'             # EsmModel + EsmTokenizer (ESM-2, ESM-1b)
    ESM_IF = 'esm_if'       # EsmForProteinFolding used as IF encoder (ESM-IF1)
    ESM_LM = 'esm_lm'       # EsmForMaskedLM (ESM-2 with LM head, function prediction)
    ESMFOLD = 'esmfold'     # EsmForProteinFolding (structure prediction)
    SAPROT = 'saprot'       # SaProt: ESM-2 backbone, needs EsmTokenizer not AutoTokenizer
    PROTTRANS = 'prottrans' # T5EncoderModel + T5Tokenizer (ProtTrans-T5 family)
    CUSTOM = 'custom'       # user-supplied loader_fn in extra dict
    OTHER = 'other'         # for any other backends not covered
    
    
    
@dataclass
class ModelConfig:
    """configuration for loading a HuggingFace model"""
    hf_id: str                          # e.g. "facebook/esm2_t33_650M_UR50D"
    task: ModelTask
    backend: ModelBackend = ModelBackend.AUTO
    layer_index: int = -1               # which layer to extract embeddings from
    half_precision: bool = False        # fp16 for large models
    extra: Dict[str, Any] = field(default_factory=dict)
    
    


class HFModelLoader:
    """
    HuggingFace model loader
    handles sequence embedding, structure prediction, and structure embedding
    """
    # registry: shorthand name -> ModelConfig
    REGISTRY: Dict[str, ModelConfig] = {

        
        # sequence embedding — esm-2 family (meta)                           
        # hf docs: https://huggingface.co/facebook/esm2_t6_8M_UR50D          
        "esm2-8m": ModelConfig(
            hf_id="facebook/esm2_t6_8M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-35m": ModelConfig(
            hf_id="facebook/esm2_t12_35M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-150m": ModelConfig(
            hf_id="facebook/esm2_t30_150M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-650m": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-3b": ModelConfig(
            hf_id="facebook/esm2_t36_3B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=True,
        ),
        "esm2-15b": ModelConfig(
            hf_id="facebook/esm2_t48_15B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=True,
        ),
        # sequence embedding,esm-2 with lm head (for function prediction)  
        # use esm_lm backend to keep the masked-lm head intact               
        "esm2-650m-lm": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.FUNCTION_PREDICTION,
            backend=ModelBackend.ESM_LM,
            layer_index=-1,
            half_precision=False,
        ),
        # sequence embedding, prot-trans family                     
        # T5EncoderModel: encoder-only, no decoder needed for embeddings      
        "prottrans-t5-xl": ModelConfig(
            hf_id="Rostlab/prot_t5_xl_uniref50",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.PROTTRANS,
            half_precision=True,
        ),
        "prottrans-t5-bfd": ModelConfig(
            hf_id="Rostlab/prot_t5_xl_bfd",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.PROTTRANS,
            half_precision=True,
        ),
        # sequence embedding, Ankh                            
        # large protein language model trained on UniRef + BFD                       
        "ankh-base": ModelConfig(
            hf_id="ElnaggarLab/ankh-base",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=False,
        ),
        "ankh-large": ModelConfig(
            hf_id="ElnaggarLab/ankh-large",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=True,
        ),
        # structure prediction — eam-fold                        
        # end-to-end folding, tokenizer is internal, returns PDB/coordinates 
        "esmfold": ModelConfig(
            hf_id="facebook/esmfold_v1",
            task=ModelTask.STRUCTURE_PREDICTION,
            backend=ModelBackend.ESMFOLD,
            half_precision=True,
        ),
        # structure-aware sequence embedding, sa-prot            #
        "saprot-35m": ModelConfig(
            hf_id="westlake-repl/SaProt_35M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
            half_precision=False,
        ),
        "saprot-650m": ModelConfig(
            hf_id="westlake-repl/SaProt_650M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
            half_precision=False,
        ),
        # inverse folding, esm-if1                                  
        # gvp-transformer; takes 3d coordinates -> sequence logits       
        "esm-if1": ModelConfig(
            hf_id="facebook/esm-if1-gvp",
            task=ModelTask.INVERSE_FOLDING,
            backend=ModelBackend.ESM_IF,
            half_precision=True,
        ),
    }
    def __init__(
        self, 
        device: Optional[torch.device] = None, 
        cache_dir: Optional[str] = None
        ) -> None:
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.cache_dir = cache_dir
        self._loaded: Dict[str, tuple] = {} # model_name -> (model, tokenizer)
        
    def load(self, name_or_id: str, config: Optional[ModelConfig] = None) -> tuple:
        """
        load by registry name ('esm2-650m') or raw hugging-face id with a config.
        returns (model, tokenizer). tokenizer is none for structure models.
        """
        if name_or_id in self._loaded:
            return self._loaded[name_or_id]
        
        cfg = config or self.REGISTRY.get(name_or_id)
        if cfg is None:
            # fallback: try treating it as a raw hf id with AUTO backend
            cfg = ModelConfig(
                hf_id=name_or_id,
                task=ModelTask.SEQUENCE_EMBEDDING,
                backend=ModelBackend.AUTO,
            )
        model, tokenizer = self._dispatch(cfg) # load the model based on its backend
        self._loaded[name_or_id] = (model, tokenizer)
        return model, tokenizer
    
    def get(self, name: str) -> Optional[tuple]:
        """return already-loaded (model, tokenizer) or raise"""
        if name not in self._loaded:
            raise KeyError(f"Model '{name}' not loaded. Call .load('{name}') first.")
        return self._loaded[name]
    
    def register(self, name: str, config: ModelConfig):
        """add a custom model to the registry"""
        self.REGISTRY[name] = config
    
    def unload(self, name: str):
        """free gpu memory"""
        if name in self._loaded:
            model, _ = self._loaded.pop(name)
            del model
            torch.cuda.empty_cache()
    
    def _dispatch(self, cfg: ModelConfig):
        """dispatch loading based on backend type"""
        loaders = {
            ModelBackend.AUTO:      self._load_auto,
            ModelBackend.ESM:       self._load_esm,
            ModelBackend.ESM_IF:    self._load_esm_if,
            ModelBackend.ESM_LM:    self._load_esm_lm,
            ModelBackend.ESMFOLD:   self._load_esmfold,
            ModelBackend.SAPROT:    self._load_saprot,
            ModelBackend.PROTTRANS: self._load_prottrans,
            ModelBackend.CUSTOM:    self._load_custom,
        }
        loader = loaders.get(cfg.backend)
        if loader is None:
            raise ValueError(f"Unknown backend: {cfg.backend}")
        return loader(cfg)
    
    def _load_auto(self, cfg: ModelConfig):
        """load using HuggingFace AutoModel and AutoTokenizer (default)"""
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm(self, cfg: ModelConfig):
        """load esm family models using fairseq's transformers (esm-1b, esm-2, etc.)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_lm(self, cfg: ModelConfig):
        """load esm-2 with the masked-lm head (for function prediction / logit extraction)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForMaskedLM.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_if(self, cfg: ModelConfig):
        """load esm-if1 (inverse folding)"""
        from transformers import EsmTokenizer as EsmTok
        tokenizer = EsmTok.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        return self._finalize(model, cfg), tokenizer

    def _load_saprot(self, cfg: ModelConfig):
        """load sa-prot (structure-aware protein transformer)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_prottrans(self, cfg: ModelConfig):
        """load prot-trans-t5 encoder"""
        tokenizer = T5Tokenizer.from_pretrained(
            cfg.hf_id, cache_dir=self.cache_dir, do_lower_case=False
        )
        model = T5EncoderModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esmfold(self, cfg: ModelConfig):
        """load esm-fold models"""
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        if cfg.extra.get("chunk_size"):
            model.trunk.set_chunk_size(cfg.extra["chunk_size"])
        return self._finalize(model, cfg), None   # no external tokenizer

    def _load_custom(self, cfg: ModelConfig):
        """load using a user-supplied loader function that returns (model, tokenizer)"""
        loader_fn = cfg.extra.get("loader_fn")
        if loader_fn is None:
            raise ValueError("CUSTOM backend requires extra['loader_fn']")
        model, tokenizer = loader_fn(cfg)
        return self._finalize(model, cfg), tokenizer
    
    def _finalize(self, model, cfg: ModelConfig):
        """set model to eval mode, move to device, and convert to half precision if specified"""
        if cfg.half_precision:
            model = model.half()
        model = model.to(self.device)
        model.eval()
        return model






class ProteinGymDataset(Dataset):
    """loads ProteinGym-style dms tables (parquet or csv)"""

    SEQ_CANDIDATES = [
        "wt_sequence",
        "wild_type_sequence",
        "wildtype",
        "sequence",
        "wt_seq",
        "reference_sequence",
    ]
    MUT_CANDIDATES = [
        "mutation",
        "mutation_string",
        "mutations",
        "variant",
        "variant_str",
        "mutant",
        "mutant_sequence",
        "mutated_sequence",
    ]
    FITNESS_CANDIDATES = [
        "score",
        "fitness",
        "measurement",
        "value",
        "ddG",
        "dG",
        "enrichment",
    ]
    ASSAY_CANDIDATES = ["assay_id", "assay", "dataset", "protein", "id", "protein_id"]
    
    def __init__(
        self,
        path: str | "pd.DataFrame",
        tokenizer: Optional[Any] = None,
        seq_col: Optional[str] = None,
        mutation_col: Optional[str] = None,
        mutant_col: Optional[str] = None,
        fitness_col: Optional[str] = None,
        assay_col: Optional[str] = None,
        max_length: Optional[int] = None,
        return_tensors: bool = False,
        strict_wt_check: bool = True,
        load_structures: bool = False,
        struct_cache_dir: Optional[str] = None,
        tokenizer_mode: str = "char",
    ) -> None:
        if pd is None:
            raise RuntimeError("pandas is required to load ProteinGym datasets")
        if isinstance(path, pd.DataFrame):
            df = path.copy()
        else:
            p = os.fspath(path)
            if p.endswith(".parquet") or p.endswith(".parq"):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        seq_col = seq_col or _find_column(df, self.SEQ_CANDIDATES)
        mutation_col = mutation_col or _find_column(df, self.MUT_CANDIDATES)
        mutant_col = mutant_col or _find_column(df, ["mutant_sequence", "mutated_sequence", "mutant_seq"]) or mutation_col
        fitness_col = fitness_col or _find_column(df, self.FITNESS_CANDIDATES)
        assay_col = assay_col or _find_column(df, self.ASSAY_CANDIDATES)
        if seq_col is None:
            raise RuntimeError("Could not infer wild-type sequence column in ProteinGym file")

        self.df = df
        self.seq_col = seq_col
        self.mutation_col = mutation_col
        self.mutant_col = mutant_col
        self.fitness_col = fitness_col
        self.assay_col = assay_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.strict_wt_check = strict_wt_check
        self.load_structures = load_structures
        self.struct_cache_dir = struct_cache_dir
        self.tokenizer_mode = tokenizer_mode
        self._records: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            wt_seq = str(row[seq_col]) if pd.notna(row[seq_col]) else None
            mut_str = None
            if mutation_col and pd.notna(row.get(mutation_col, None)):
                mut_str = str(row[mutation_col])
            mutant_seq = None
            if mutant_col and pd.notna(row.get(mutant_col, None)):
                mutant_seq = str(row[mutant_col])
            fitness = float(row[fitness_col]) if (fitness_col and pd.notna(row.get(fitness_col, None))) else None
            assay_id = row[assay_col] if (assay_col and pd.notna(row.get(assay_col, None))) else None
            parsed = None
            if mut_str:
                try:
                    parsed = parse_mutation_string(mut_str)
                except Exception:
                    parsed = None
            if mutant_seq is None and parsed is not None and wt_seq is not None:
                try:
                    from .tokenizers import apply_mutations
                    mutant_seq = apply_mutations(wt_seq, parsed)
                except Exception:
                    mutant_seq = None
            if parsed is None and wt_seq is not None and mutant_seq is not None and len(wt_seq) == len(mutant_seq):
                diffs = []
                for i, (a, b) in enumerate(zip(wt_seq, mutant_seq)):
                    if a != b:
                        diffs.append(f"{a}{i+1}{b}")
                if diffs:
                    try:
                        parsed = parse_mutation_string(":".join(diffs))
                    except Exception:
                        parsed = None
            rec = {
                "wt_sequence": wt_seq,
                "mutation_string": mut_str,
                "mutant_sequence": mutant_seq,
                "parsed_mutation": parsed,
                "fitness": fitness,
                "assay_id": assay_id,
            }
            self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = dict(self._records[idx])
        if self.load_structures:
            try:
                rec["wt_structure"] = load_structure(rec["wt_sequence"]) if rec.get("wt_sequence") else None
                rec["mutant_structure"] = load_structure(rec["mutant_sequence"]) if rec.get("mutant_sequence") else None
            except Exception as e:
                log.warning("Could not load structure for idx %d: %s", idx, e)
        if self.tokenizer is not None:
            try:
                if isinstance(self.tokenizer, str):
                    rec["input_wt"] = tokenise_sequence(rec["wt_sequence"], model_name=self.tokenizer, return_tensors=self.return_tensors)
                    if rec.get("mutation_string"):
                        from .tokenizers import ProteinTokenizer
                        ptok = ProteinTokenizer(mode=self.tokenizer_mode)
                        rec["input_mutant"] = ptok.encode_mutation_string(rec["wt_sequence"], rec["mutation_string"], return_tensors=self.return_tensors)
                    else:
                        rec["input_mutant"] = None
                else:
                    tok = self.tokenizer
                    rec["input_wt"] = tok.encode_sequence(rec["wt_sequence"], max_length=self.max_length, padding=False, truncation=True, return_tensors=self.return_tensors)
                    if rec.get("mutation_string"):
                        rec["input_mutant"] = tok.encode_mutation_string(rec["wt_sequence"], rec["mutation_string"], max_length=self.max_length, padding=False, truncation=True, return_tensors=self.return_tensors, strict_wt_check=self.strict_wt_check)
                    else:
                        rec["input_mutant"] = None
            except Exception as e:
                log.warning("Tokenization failed for idx %d: %s", idx, e)

        if rec.get("mutation_string"):
            try:
                rec["action"] = encode_mutation(rec["mutation_string"])
            except Exception:
                rec["action"] = None
        else:
            rec["action"] = None

        return rec




class TsuboyamaDataset(ProteinGymDataset):
    """loads Tsuboyama stability datasets (wrapper around ProteinGymDataset)"""

    def __init__(self, path: str | "pd.DataFrame", **kwargs) -> None:
        super().__init__(path, **kwargs)



class FitnessTransitionDataset(Dataset):
    """Constructs (state, action, next_state, reward) tuples, each transition represents a single mutation"""
    def __init__(self, base_dataset: ProteinGymDataset) -> None:
        if not isinstance(base_dataset, ProteinGymDataset):
            raise TypeError("FitnessTransitionDataset expects a ProteinGymDataset")
        self.base = base_dataset
        # build transitions as (wt, action, mutant, reward, assay_id)
        self._transitions: List[Dict[str, Any]] = []
        for i in range(len(self.base)):
            s = self.base[i]
            if s.get("wt_sequence") and s.get("mutant_sequence"):
                self._transitions.append({
                    "s_t": {"sequence": s["wt_sequence"]},
                    "action": s.get("action"),
                    "s_t1": {"sequence": s["mutant_sequence"]},
                    "reward": s.get("fitness"),
                    "assay_id": s.get("assay_id"),
                })

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._transitions[idx]



class CustomAssayDataset(ProteinGymDataset):
    """dataset for user-uploaded csv files with `sequence` and `fitness` columns"""
    def __init__(self, path: str | "pd.DataFrame", seq_col: str = "sequence", fitness_col: str = "fitness", **kwargs) -> None:
        super().__init__(path, seq_col=seq_col, fitness_col=fitness_col, **kwargs)
        
        

def _find_column(df: "pd.DataFrame", candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None




def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _resolve_root(cfg: dict) -> Path:
    root = Path(cfg["data_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root

def _progress_wrap(iterable, *, total: int, desc: str, show: bool):
    """wrap an iterable with tqdm if available and requested"""
    if show and HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="B",
                    unit_scale=True, unit_divisor=1024)
    return iterable

def _http_download(
    url: str,
    dest: Path,
    *,
    show_progress: bool = True,
    chunk_size: int = 8192,
    max_retries: int = 3,
    dry_run: bool = False,
) -> None:
    """download a file over https with resume-friendly streaming and retries"""
    if dry_run:
        log.info("[DRY-RUN] Would download %s → %s", url, dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                chunks = resp.iter_content(chunk_size=chunk_size)
                wrapped = _progress_wrap(
                    chunks, total=total,
                    desc=dest.name, show=show_progress
                )
                with open(dest, "wb") as fh:
                    for chunk in wrapped:
                        if chunk:
                            fh.write(chunk)
            log.info("✓  Saved %s", dest)
            return
        except (requests.RequestException, IOError) as exc:
            log.warning("Attempt %d/%d failed for %s: %s",
                        attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)   # exponential back-off
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts"
                ) from exc
                

def _s3_download(
    bucket: str,
    key: str,
    dest: Path,
    *,
    region: str = "us-east-2",
    show_progress: bool = True,
    dry_run: bool = False,
) -> None:
    """download a single object from a public s3 bucket (no credentials)"""
    if dry_run:
        log.info("[DRY-RUN] Would download s3://%s/%s → %s", bucket, key, dest)
        return

    if not HAS_BOTO3:
        # fall back to https public url if boto3 not available
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        log.warning("boto3 not installed – falling back to HTTPS: %s", url)
        _http_download(url, dest, show_progress=show_progress)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotocoreConfig(signature_version=UNSIGNED),
    )

    # get object size for progress bar
    head = s3.head_object(Bucket=bucket, Key=key)
    total_bytes = head["ContentLength"]

    log.info("Downloading s3://%s/%s  (%.1f MB)", bucket, key,
             total_bytes / 1e6)

    if show_progress and HAS_TQDM:
        bar = tqdm(total=total_bytes, desc=dest.name, unit="B",
                   unit_scale=True, unit_divisor=1024)
        callback = lambda bytes_transferred: bar.update(bytes_transferred)  # noqa: e731
    else:
        bar = None
        callback = None

    try:
        s3.download_file(
            bucket, key, str(dest),
            Callback=callback,
        )
    finally:
        if bar:
            bar.close()

    log.info("✓  Saved %s", dest)

def _unzip(archive: Path, dest_dir: Path) -> None:
    log.info("Unzipping %s → %s", archive.name, dest_dir)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    log.info("✓  Extracted %s", archive.name)

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()




# ProteinGym downloader
def download_proteingym(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """download selected ProteinGym subsets from s3 or hugging-face"""
    pg_cfg = cfg["proteingym"]
    if not pg_cfg.get("enabled", True):
        log.info("ProteinGym download disabled – skipping.")
        return

    out_dir = root / pg_cfg.get("output_subdir", "proteingym")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)

    source = pg_cfg.get("source", "s3")
    bucket = pg_cfg.get("s3_bucket", "proteingym")
    region = pg_cfg.get("s3_region", "us-east-2")

    subsets: dict = pg_cfg.get("subsets", {})

    for subset_name, enabled in subsets.items():
        if not enabled:
            continue
        if subset_name not in PROTEINGYM_S3_KEYS:
            log.warning("Unknown subset '%s' – skipping.", subset_name)
            continue

        s3_key, filename = PROTEINGYM_S3_KEYS[subset_name]
        dest = out_dir / filename

        if skip and dest.exists():
            log.info("⏭  %s already exists – skipping.", dest.name)
            continue

        log.info("━━  Downloading ProteinGym / %s", subset_name)

        if source == "s3":
            _s3_download(bucket, s3_key, dest,
                         region=region, show_progress=prog, dry_run=dry_run)
        elif source == "huggingface":
            _download_proteingym_hf(subset_name, dest, dry_run=dry_run)
        else:
            raise ValueError(f"Unknown source '{source}'. Use 's3' or 'huggingface'.")

    # reference/metadata files
    if pg_cfg.get("reference_files", True):
        ref_dir = out_dir / "reference_files"
        ref_dir.mkdir(exist_ok=True)
        for fname, url in PROTEINGYM_REF_URLS.items():
            dest = ref_dir / fname
            if skip and dest.exists():
                log.info("⏭  %s already exists – skipping.", fname)
                continue
            log.info("━━  Downloading reference file: %s", fname)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries,
                           dry_run=dry_run)


def _download_proteingym_hf(subset_name: str, dest: Path,
                             dry_run: bool = False) -> None:
    """alternative: stream a ProteinGym split from hugging-face and save as parquet"""
    if dry_run:
        log.info("[DRY-RUN] Would download HF split %s → %s", subset_name, dest)
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Hugging Face 'datasets' library not installed.\n"
            "Run: pip install datasets"
        )

    log.info("Streaming ProteinGym/%s from Hugging Face...", subset_name)
    ds = load_dataset("OATML-Markslab/ProteinGym_v1",
                      name=subset_name, split="train")
    df = ds.to_pandas()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    log.info("✓  Saved %s  (%d rows)", dest, len(df))




# Tsuboyama downloader
def download_tsuboyama(cfg: dict, root: Path, dry_run: bool = False) -> None:
    """download selected files from the tsuboyama 2023 zenodo record"""
    ts_cfg = cfg["tsuboyama"]
    if not ts_cfg.get("enabled", True):
        log.info("Tsuboyama download disabled – skipping.")
        return

    out_dir = root / ts_cfg.get("output_subdir", "tsuboyama")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg  = cfg.get("download", {})
    skip    = dl_cfg.get("skip_existing", True)
    prog    = dl_cfg.get("show_progress", True)
    retries = dl_cfg.get("max_retries", 3)
    chunk   = dl_cfg.get("chunk_size", 8192)

    base_url = ts_cfg.get("zenodo_base_url",
                           "https://zenodo.org/records/7992926/files")
    do_unzip = ts_cfg.get("unzip", True)

    files_cfg: dict = ts_cfg.get("files", {})

    for key, enabled in files_cfg.items():
        if not enabled:
            continue
        if key not in TSUBOYAMA_FILES:
            log.warning("Unknown Tsuboyama file key '%s' – skipping.", key)
            continue

        filename = TSUBOYAMA_FILES[key]
        dest     = out_dir / filename
        url      = f"{base_url}/{filename}?download=1"

        if skip and dest.exists():
            log.info("⏭  %s already exists – skipping.", filename)
        else:
            log.info("━━  Downloading Tsuboyama / %s", filename)
            _http_download(url, dest, show_progress=prog,
                           chunk_size=chunk, max_retries=retries,
                           dry_run=dry_run)

        # unzip if it is an archive and unzip is requested
        if do_unzip and filename.endswith(".zip") and (dry_run or dest.exists()):
            unzip_dir = out_dir / filename.replace(".zip", "")
            if skip and unzip_dir.exists():
                log.info("⏭  %s already extracted – skipping.", unzip_dir.name)
            elif not dry_run:
                _unzip(dest, unzip_dir)
            else:
                log.info("[DRY-RUN] Would unzip %s → %s", dest.name, unzip_dir)





def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download raw data for ProteinDreamer."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/protein_dreamer/download_config.yaml",
        help="Path to download_config.yaml (default: configs/protein_dreamer/download_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without making any network calls.",
    )
    parser.add_argument(
        "--only",
        choices=["proteingym", "tsuboyama"],
        default=None,
        help="Download only one data source.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg  = _load_config(cfg_path)
    root = _resolve_root(cfg)
    log.info("Data root: %s", root.resolve())

    if args.dry_run:
        log.info("DRY-RUN mode – no files will be written.")

    if args.only in (None, "proteingym"):
        download_proteingym(cfg, root, dry_run=args.dry_run)

    if args.only in (None, "tsuboyama"):
        download_tsuboyama(cfg, root, dry_run=args.dry_run)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Download complete. Data saved under: %s", root.resolve())







def _is_tensor_like(x: Any) -> bool:
    return hasattr(x, "numpy") or hasattr(x, "detach") or isinstance(x, (list, tuple))


def make_collate_fn(dataset: Optional[Any] = None, pad_token_id: Optional[int] = None) -> Callable:
    """create a collate function that pads token sequences and stacks tensors"""
    if pad_token_id is None and dataset is not None:
        tok = getattr(dataset, "tokenizer", None)
        if tok is not None and hasattr(tok, "pad_token_id"):
            try:
                pad_token_id = int(tok.pad_token_id)
            except Exception:
                pad_token_id = 0
    pad_token_id = 0 if pad_token_id is None else pad_token_id

    def _pad_1d(tensors: List["torch.Tensor"], pad_value: int = 0) -> "torch.Tensor":
        lengths = [t.size(0) for t in tensors]
        max_len = max(lengths) if lengths else 0
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    def _collate_field(values: List[Any]) -> Any:
        if all(v is None for v in values):
            return None

        if all(hasattr(v, "dim") for v in values if v is not None):
            tensors = [v.squeeze(0) if (v is not None and getattr(v, "dim", lambda: 1)() == 2 and v.size(0) == 1) else v for v in values if v is not None]
            if all(t.dim() == 1 for t in tensors):
                return _pad_1d(tensors, pad_value=pad_token_id)
            try:
                return torch.stack(tensors)
            except Exception:
                max_last = max(t.size(-1) for t in tensors)
                out_shape = (len(tensors),) + tuple(tensors[0].size()[:-1]) + (max_last,)
                out = tensors[0].new_zeros(out_shape)
                for i, t in enumerate(tensors):
                    sl = t.size(-1)
                    out[i, ..., :sl] = t
                return out
        if all(isinstance(v, (list, tuple)) for v in values if v is not None):
            tensors = [torch.tensor(v, dtype=torch.long) for v in values if v is not None]
            return _pad_1d(tensors, pad_value=pad_token_id)
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            if any(isinstance(v, float) for v in values if v is not None):
                return torch.tensor([float(v) if v is not None else float('nan') for v in values], dtype=torch.float)
            return torch.tensor([int(v) if v is not None else 0 for v in values], dtype=torch.long)
        return [v for v in values]

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        keys = set().union(*(b.keys() for b in batch))
        for k in keys:
            vals = [b.get(k, None) for b in batch]
            if any(isinstance(v, dict) for v in vals if v is not None):
                dict_list = [v if isinstance(v, dict) else {} for v in vals]
                nested_keys = set().union(*(d.keys() for d in dict_list))
                nested_out: Dict[str, Any] = {}
                for nk in nested_keys:
                    nested_vals = [d.get(nk, None) for d in dict_list]
                    nested_out[nk] = _collate_field(nested_vals)
                out[k] = nested_out
            else:
                out[k] = _collate_field(vals)
        return out
    return collate_fn



def make_dataloader(
    dataset: Any,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
):
    """construct a PyTorch DataLoader"""
    if collate_fn is None:
        collate_fn = make_collate_fn(dataset)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=collate_fn
        )



def make_dataloaders(
    train_dataset: Optional[Any],
    val_dataset: Optional[Any] = None,
    test_dataset: Optional[Any] = None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = False,
):
    """build train/val/test DataLoaders"""
    loaders = {}
    if train_dataset is not None:
        train_collate = make_collate_fn(train_dataset)
        loaders["train"] = make_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            collate_fn=train_collate
            )
    else:
        loaders["train"] = None
    if val_dataset is not None:
        val_collate = make_collate_fn(val_dataset)
        vb = val_batch_size or batch_size
        loaders["val"] = make_dataloader(
            val_dataset, 
            batch_size=vb, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            collate_fn=val_collate
            )
    else:
        loaders["val"] = None

    if test_dataset is not None:
        test_collate = make_collate_fn(test_dataset)
        tb = test_batch_size or batch_size
        loaders["test"] = make_dataloader(
            test_dataset, 
            batch_size=tb, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            collate_fn=test_collate
            )
    else:
        loaders["test"] = None
    return loaders


__all__ = ["make_collate_fn", "make_dataloader", "make_dataloaders"]




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
  from .tokenizers import parse_mutation_string as _pms
  return _pms(mut_str)


def encode_mutation(mutation: Any) -> Dict[str, torch.Tensor]:
  """encode (tokenize) a mutation specification (string or structured) into tensors for model input"""
  from .tokenizers import MutationRecord, ParsedMutation
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