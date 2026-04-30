from __future__ import annotations
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
                    from .tokenizers.mutation_tokenizer import apply_mutations
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
                        from .tokenizers.mutation_tokenizer import ProteinTokenizer
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