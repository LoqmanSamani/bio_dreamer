"""
parses ProteinGym / Tsuboyama mutation strings and produces:
  1. structured MutationRecord objects (wt_aa, position, mut_aa, …)
  2. encoded token representations via an underlying sequence tokenizer

supported mutation string formats
----------------------------------
  single substitution    "A42G"           wt=A, pos=42, mut=G
  multi-substitution     "A42G:L100V"     colon-separated
  tsuboyama dg format    "A42G"           identical format, different fitness scale
  deletion (indel)       "A42-"           mut_aa = '-' (skip for single-sub models)
  insertion (indel)      "42ins3"         flagged, not parsed into aa tokens

the ProteinTokenizer wraps either CharTokenizer or KmerTokenizer or BPETokenizer and exposes:
  - encode_sequence()          → standard sequence encoding
  - encode_mutation_string()   → encode the *mutated* sequence with [MUT] flags
  - parse_mutation()           → returns MutationRecord(s)
  - encode_with_organism()     → prepend organism token before [CLS]
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal
import torch
from ..core.tokenizers import BaseProteinTokenizer
from .char_tokenizer import CharTokenizer
from .kmer_tokenizer import KmerTokenizer
from .bpe_tokenizer import BPETokenizer





# tokenizer mode type 
TokenizerMode = Literal["char", "kmer", "bpe"]
# regex for parsing mutation sub-strings
_SINGLE_SUB_RE = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z\-*])$")

# insertion flag: e.g. "42ins3"  (not decoded into aa tokens)
_INSERTION_RE = re.compile(r"^\d+ins\d+$", re.IGNORECASE)




class ProteinTokenizer:
    """
    wraps CharTokenizer, KmerTokenizer, or BPETokenizer and adds:
      - mutation string parsing
      - mutated-sequence encoding with automatic [MUT] position flagging
      - organism token prepending
    """
    def __init__(
        self,
        mode: TokenizerMode = "char",
        k: int = 3,
        stride: int | None = None,
        bpe_vocab_size: int = 200,
        bpe_merges: list[tuple[str, str]] | None = None,
        organism_tokens: list[str] | None = None,
        add_special_tokens: bool = True,
        include_ambiguous: bool = True,
    ) -> None:
        self.mode = mode
        self.organism_tokens = organism_tokens or []

        if mode == "char":
            self._tok: BaseProteinTokenizer = CharTokenizer(
                add_special_tokens=add_special_tokens,
                organism_tokens=organism_tokens,
                include_ambiguous=include_ambiguous,
            )
        elif mode == "kmer":
            self._tok = KmerTokenizer(
                k=k,
                stride=stride,
                add_special_tokens=add_special_tokens,
                organism_tokens=organism_tokens,
                include_ambiguous=include_ambiguous,
            )
        elif mode == "bpe":
            self._tok = BPETokenizer(
                vocab_size=bpe_vocab_size,
                include_ambiguous=include_ambiguous,
                add_special_tokens=add_special_tokens,
                organism_tokens=organism_tokens,
                merges=bpe_merges,
            )
        else:
            raise ValueError(
                f"Unknown tokenizer mode '{mode}'. Use 'char', 'kmer', or 'bpe'."
            )

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def mask_token_id(self) -> int:
        return self._tok.mask_token_id

    def get_vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_sequence(
        self,
        sequence: str,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: bool = False,
        mutation_positions: list[int] | None = None,
    ) -> dict:
        """encode a raw amino acid sequence. thin wrapper around underlying tokenizer"""
        return self._tok.encode(
            sequence,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            mutation_positions=mutation_positions,
        )

    def batch_encode_sequences(
        self,
        sequences: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = False,
        mutation_positions: list[list[int]] | None = None,
    ) -> dict:
        return self._tok.batch_encode(
            sequences,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            mutation_positions=mutation_positions,
        )

    def parse_mutation(self, mutation_str: str) -> ParsedMutation:
        """parse a mutation string into a ParsedMutation (no encoding)"""
        return parse_mutation_string(mutation_str)

    def encode_mutation_string(
        self,
        wt_sequence: str,
        mutation_str: str,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: bool = False,
        strict_wt_check: bool = True,
    ) -> dict:
        """
        parse mutation_str, apply it to wt_sequence, then encode the
        resulting mutated sequence with mutation positions flagged
        """
        parsed = parse_mutation_string(mutation_str)

        try:
            mutated_seq = apply_mutations(wt_sequence, parsed)
        except ValueError as e:
            if strict_wt_check:
                raise
            # soft mode: encode the wt sequence without flagging positions
            mutated_seq = wt_sequence

        out = self._tok.encode(
            mutated_seq,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            mutation_positions=parsed.positions_0,
        )
        out["parsed_mutation"] = parsed
        return out

    def batch_encode_mutations(
        self,
        wt_sequences: list[str],
        mutation_strings: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = False,
        strict_wt_check: bool = True,
    ) -> dict:
        """batch version of encode_mutation_string"""
        encoded_list = []
        parsed_list  = []

        for wt_seq, mut_str in zip(wt_sequences, mutation_strings):
            enc = self.encode_mutation_string(
                wt_seq, mut_str,
                padding=False,
                truncation=truncation,
                max_length=max_length,
                return_tensors=False,
                strict_wt_check=strict_wt_check,
            )
            parsed_list.append(enc.pop("parsed_mutation"))
            encoded_list.append(enc)
        # pad to batch max
        batch_max = max(len(e["input_ids"]) for e in encoded_list)
        if max_length is not None:
            batch_max = min(batch_max, max_length)
        if padding:
            pad_id = self._tok.pad_token_id
            for e in encoded_list:
                pad_len = batch_max - len(e["input_ids"])
                e["input_ids"]      += [pad_id] * pad_len
                e["attention_mask"] += [0]       * pad_len
                e["mutation_mask"]  += [0]       * pad_len
        if return_tensors:
            return {
                "input_ids":       torch.tensor(
                    [e["input_ids"]      for e in encoded_list], dtype=torch.long),
                "attention_mask":  torch.tensor(
                    [e["attention_mask"] for e in encoded_list], dtype=torch.long),
                "mutation_mask":   torch.tensor(
                    [e["mutation_mask"]  for e in encoded_list], dtype=torch.long),
                "parsed_mutations": parsed_list,
            }
        return {
            "input_ids":       [e["input_ids"]      for e in encoded_list],
            "attention_mask":  [e["attention_mask"] for e in encoded_list],
            "mutation_mask":   [e["mutation_mask"]  for e in encoded_list],
            "parsed_mutations": parsed_list,
        }

    def encode_with_organism(
        self,
        sequence: str,
        organism: str,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: bool = False,
        mutation_positions: list[int] | None = None,
    ) -> dict:
        """encode sequence and prepend an organism token before [CLS]"""
        org_id = self._tok.get_organism_token_id(organism)
        
        out = self._tok.encode(
            sequence,
            max_length=(max_length - 1) if max_length else None,
            padding=False,
            truncation=truncation,
            return_tensors=False,
            mutation_positions=mutation_positions,
        )
        # prepend organism token
        out["input_ids"]      = [org_id] + out["input_ids"]
        out["attention_mask"] = [1]      + out["attention_mask"]
        out["mutation_mask"]  = [0]      + out["mutation_mask"]

        if padding and max_length is not None:
            pad_len = max_length - len(out["input_ids"])
            out["input_ids"]      += [self._tok.pad_token_id] * pad_len
            out["attention_mask"] += [0]                       * pad_len
            out["mutation_mask"]  += [0]                       * pad_len

        if return_tensors:
            return {k: torch.tensor(v, dtype=torch.long)
                    if isinstance(v, list) else v
                    for k, v in out.items()}
        return out

    # bpe-specific helpers
    def train_bpe(
        self,
        sequences: list[str],
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """train bpe merge rules from a corpus of sequences"""
        if self.mode != "bpe":
            raise RuntimeError(
                "train_bpe() is only available when mode='bpe'. "
                f"Current mode: '{self.mode}'."
            )
        assert isinstance(self._tok, BPETokenizer)
        self._tok.train(sequences, min_frequency=min_frequency, verbose=verbose)

    def save_bpe(self, path: str) -> None:
        """save the bpe vocabulary/merge table to a json file"""
        if self.mode != "bpe":
            raise RuntimeError("save_bpe() requires mode='bpe'.")
        assert isinstance(self._tok, BPETokenizer)
        self._tok.save(path)

    @classmethod
    def load_bpe(
        cls,
        path: str,
        organism_tokens: list[str] | None = None,
    ) -> ProteinTokenizer:
        """load a previously saved bpe tokenizer and return a ProteinTokenizer"""
        saved = BPETokenizer.load(path)
        return cls(
            mode="bpe",
            bpe_vocab_size=saved.target_vocab_size,
            bpe_merges=saved.merges,
            organism_tokens=organism_tokens or saved.organism_tokens,
            add_special_tokens=saved.add_special_tokens,
        )

    def __repr__(self) -> str:
        return (
            f"UnifiedTokenizer(mode='{self.mode}', "
            f"vocab_size={self.vocab_size}, "
            f"organism_tokens={self.organism_tokens})"
        )
        
            

@dataclass
class MutationRecord:
    """represents a single amino acid substitution or indel from a mutation string"""
    wt_aa:    str          # wild-type amino acid (single letter)
    position: int          # 1-based position (as in ProteinGym)
    mut_aa:   str          # mutant amino acid
    raw:      str = ""     # original string, e.g. "A42G"

    @property
    def position_0(self) -> int:
        """0-based position for use with python strings"""
        return self.position - 1

    @property
    def is_synonymous(self) -> bool:
        return self.wt_aa == self.mut_aa

    @property
    def is_deletion(self) -> bool:
        return self.mut_aa == "-"

    def __str__(self) -> str:
        return self.raw or f"{self.wt_aa}{self.position}{self.mut_aa}"

@dataclass
class ParsedMutation:
    """container for one or more MutationRecords from a mutation string"""
    records:       list[MutationRecord]
    raw_string:    str
    n_mutations:   int = field(init=False)
    has_indels:    bool = field(init=False)

    def __post_init__(self):
        self.n_mutations = len(self.records)
        self.has_indels  = any(r.is_deletion for r in self.records)

    @property
    def is_single(self) -> bool:
        return self.n_mutations == 1

    @property
    def positions_0(self) -> list[int]:
        """0-based positions of all mutations"""
        return [r.position_0 for r in self.records]

def parse_mutation_string(mut_str: str) -> ParsedMutation:
    """parse a ProteinGym-style mutation string into a ParsedMutation"""
    parts = [p.strip() for p in mut_str.split(":")]
    records: list[MutationRecord] = []
    
    for part in parts:
        if _INSERTION_RE.match(part):
            # insertions: we record them as a deletion placeholder
            records.append(MutationRecord(
                wt_aa="-", position=0, mut_aa="ins", raw=part
            ))
            continue

        m = _SINGLE_SUB_RE.match(part)
        if not m:
            raise ValueError(
                f"Cannot parse mutation token '{part}' in '{mut_str}'. "
                "Expected format: <WT_AA><1-based position><MUT_AA> "
                "e.g. 'A42G' or 'A42G:L100V'."
            )
        wt, pos_str, mut = m.group(1), m.group(2), m.group(3)
        records.append(MutationRecord(
            wt_aa=wt.upper(),
            position=int(pos_str),
            mut_aa=mut.upper(),
            raw=part,
        ))
    return ParsedMutation(records=records, raw_string=mut_str)

def apply_mutations(wt_sequence: str, parsed: ParsedMutation) -> str:
    """apply parsed mutations to a wild-type sequence string"""
    seq = list(wt_sequence.upper())
    for rec in parsed.records:
        if rec.position == 0:
            continue  
        idx = rec.position_0
        if idx >= len(seq):
            raise IndexError(
                f"Mutation position {rec.position} out of range "
                f"for sequence of length {len(seq)}."
            )
        if seq[idx] != rec.wt_aa and rec.wt_aa not in ("*", "-"):
            raise ValueError(
                f"WT mismatch at position {rec.position}: "
                f"expected '{rec.wt_aa}', found '{seq[idx]}'."
            )
        if rec.is_deletion:
            seq[idx] = "" 
        else:
            seq[idx] = rec.mut_aa
    return "".join(seq)