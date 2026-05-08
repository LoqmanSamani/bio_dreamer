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
  - encode_sequence()          -> standard sequence encoding
  - encode_mutation_string()   -> encode the *mutated* sequence with [MUT] flags
  - parse_mutation()           -> returns MutationRecord(s)
  - encode_with_organism()     -> prepend organism token before [CLS]
"""
from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any

import torch

from ..core.tokenizers import BaseProteinTokenizer
from biodreamer.protein_dreamer.config import ProteinDreamerConfig

logger = logging.getLogger(__name__)

TokenizerMode = Literal["char", "kmer", "bpe"]

_SINGLE_SUB_RE = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z\-*])$")
_INSERTION_RE = re.compile(r"^\d+ins\d+$", re.IGNORECASE)





class ProteinTokenizer:
    """
    wraps CharTokenizer, KmerTokenizer, or BPETokenizer and adds:
      - mutation string parsing
      - mutated-sequence encoding with automatic [MUT] position flagging
      - organism token prepending
    """
    def __init__(self, config: Any = None) -> None:
        if config is None:
            config = ProteinDreamerConfig().default()["tokenizer"]
        mode             = config.get("mode", "char")
        k                = config.get("k", 3)
        stride           = config.get("stride", None)
        bpe_vocab_size   = config.get("bpe_vocab_size", 200)
        bpe_merges       = config.get("bpe_merges", None)
        organism_tokens  = config.get("organism_tokens", None)
        add_special_tokens = config.get("add_special_tokens", True)
        include_ambiguous  = config.get("include_ambiguous", True)
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
        """parse mutation_str, apply it to wt_sequence, then encode with mutation positions flagged"""
        parsed = parse_mutation_string(mutation_str)

        try:
            mutated_seq = apply_mutations(wt_sequence, parsed)
        except ValueError:
            if strict_wt_check:
                raise
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

    def train_bpe(
        self,
        sequences: list[str],
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        if self.mode != "bpe":
            raise RuntimeError(
                "train_bpe() is only available when mode='bpe'. "
                f"Current mode: '{self.mode}'."
            )
        self._tok.train(sequences, min_frequency=min_frequency, verbose=verbose)

    def save_bpe(self, path: str) -> None:
        if self.mode != "bpe":
            raise RuntimeError("save_bpe() requires mode='bpe'.")
        self._tok.save(path)

    @classmethod
    def load_bpe(
        cls,
        path: str,
        organism_tokens: list[str] | None = None,
    ) -> ProteinTokenizer:
        saved = BPETokenizer.load(path)
        return cls({
            "mode":               "bpe",
            "bpe_vocab_size":     saved.target_vocab_size,
            "bpe_merges":         saved.merges,
            "organism_tokens":    organism_tokens or saved.organism_tokens,
            "add_special_tokens": saved.add_special_tokens,
        })

    def __repr__(self) -> str:
        return (
            f"ProteinTokenizer(mode='{self.mode}', "
            f"vocab_size={self.vocab_size}, "
            f"organism_tokens={self.organism_tokens})"
        )


@dataclass
class MutationRecord:
    """represents a single amino acid substitution or indel from a mutation string"""
    wt_aa:    str
    position: int
    mut_aa:   str
    raw:      str = ""

    @property
    def position_0(self) -> int:
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
        return [r.position_0 for r in self.records]


class BPETokenizer(BaseProteinTokenizer):
    """learned bpe tokenizer for amino acid sequences"""
    def __init__(
        self,
        vocab_size: int = 200,
        include_ambiguous: bool = True,
        add_special_tokens: bool = True,
        organism_tokens: list[str] | None = None,
        merges: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(
            add_special_tokens=add_special_tokens,
            organism_tokens=organism_tokens,
        )
        self.target_vocab_size = vocab_size
        self.include_ambiguous = include_ambiguous
        self.merges: list[tuple[str, str]] = merges or []
        self._build_seed_vocab()
        if self.merges:
            self._apply_merges_to_vocab(self.merges)

    def _build_seed_vocab(self) -> None:
        seed = (
            self.SPECIAL_TOKENS
            + self.organism_tokens
            + self.AMINO_ACIDS
            + (self.AMBIGUOUS_AA if self.include_ambiguous else [])
        )
        self._register_vocab(seed)

    def _apply_merges_to_vocab(self, merges: list[tuple[str, str]]) -> None:
        for a, b in merges:
            merged = a + b
            if merged not in self.token2id:
                self._register_vocab([merged])

    def train(
        self,
        sequences: list[str],
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        working_vocab: dict[tuple[str, ...], int] = Counter(
            tuple(_word_to_chars(seq))
            for seq in sequences
            if seq.strip()
        )
        n_seed     = self.vocab_size
        max_merges = self.target_vocab_size - n_seed

        if max_merges <= 0:
            if verbose:
                print(f"[BPE] vocab_size={self.target_vocab_size} already reached "
                      f"by seed vocabulary ({n_seed} tokens). No merges needed.")
            return

        self.merges = []
        for step in range(max_merges):
            stats = _get_pair_stats(working_vocab)
            if not stats:
                break
            stats = Counter({k: v for k, v in stats.items() if v >= min_frequency})
            if not stats:
                break
            best_pair = max(stats, key=lambda p: (stats[p], p))
            working_vocab = _merge_vocab(working_vocab, best_pair)
            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            self._register_vocab([merged_token])
            if verbose and (step + 1) % 50 == 0:
                print(f"[BPE] step {step+1:4d} | vocab={self.vocab_size} | "
                      f"merged '{best_pair[0]}'+'{best_pair[1]}' "
                      f"→ '{merged_token}' (freq={stats[best_pair]})")
        if verbose:
            print(f"[BPE] Training done. Final vocab size: {self.vocab_size} "
                  f"({len(self.merges)} merges learned).")

    def tokenize(self, sequence: str) -> list[str]:
        if not self.merges:
            return _word_to_chars(sequence)
        word = _word_to_chars(sequence.upper())
        for pair in self.merges:
            word = _apply_merge(word, pair)
        return word

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.unk_token_id
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = set()
        if skip_special_tokens:
            special_ids = {self.token2id[t] for t in self.SPECIAL_TOKENS
                           if t in self.token2id}
            special_ids |= {self.token2id[t] for t in self.organism_tokens
                            if t in self.token2id}

        return "".join(
            self.id2token[i]
            for i in ids
            if i in self.id2token and i not in special_ids
        )

    def _build_mutation_mask(
        self,
        tokens: list[str],
        mutation_positions: list[int] | None,
    ) -> list[int]:
        mask = [0] * len(tokens)
        if not mutation_positions:
            return mask
        spans: list[tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            spans.append((cursor, cursor + len(tok) - 1))
            cursor += len(tok)
        for pos in mutation_positions:
            for i, (start, end) in enumerate(spans):
                if start <= pos <= end:
                    mask[i] = 1
        return mask

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type":               "BPETokenizer",
            "target_vocab_size":  self.target_vocab_size,
            "include_ambiguous":  self.include_ambiguous,
            "add_special_tokens": self.add_special_tokens,
            "organism_tokens":    self.organism_tokens,
            "merges":             [list(m) for m in self.merges],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        if data.get("type") != "BPETokenizer":
            raise ValueError(f"File {path} does not contain a BPETokenizer.")
        return cls(
            vocab_size=data["target_vocab_size"],
            include_ambiguous=data["include_ambiguous"],
            add_special_tokens=data["add_special_tokens"],
            organism_tokens=data.get("organism_tokens"),
            merges=[tuple(m) for m in data["merges"]],
        )

    def __repr__(self) -> str:
        trained = f"{len(self.merges)} merges" if self.merges else "untrained"
        return (
            f"BPETokenizer(target_vocab_size={self.target_vocab_size}, "
            f"vocab_size={self.vocab_size}, {trained})"
        )


class KmerTokenizer(BaseProteinTokenizer):
    """k-mer amino acid tokenizer with configurable k and stride"""
    def __init__(
        self,
        k: int = 3,
        stride: int | None = None,
        pad_incomplete: bool = False,
        add_special_tokens: bool = True,
        organism_tokens: list[str] | None = None,
        include_ambiguous: bool = True,
    ) -> None:
        super().__init__(
            add_special_tokens=add_special_tokens,
            organism_tokens=organism_tokens,
        )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k               = k
        self.stride          = stride if stride is not None else 1
        self.pad_incomplete  = pad_incomplete
        self.include_ambiguous = include_ambiguous
        self._build_vocab()

    def _build_vocab(self) -> None:
        aa = self.AMINO_ACIDS + (self.AMBIGUOUS_AA if self.include_ambiguous else [])
        kmers = sorted(
            "".join(combo)
            for combo in itertools.product(aa, repeat=self.k)
        )
        tokens = self.SPECIAL_TOKENS + self.organism_tokens + kmers
        self._register_vocab(tokens)

    def tokenize(self, sequence: str) -> list[str]:
        seq = sequence.upper()
        L   = len(seq)
        tokens: list[str] = []

        pos = 0
        while pos + self.k <= L:
            tokens.append(seq[pos : pos + self.k])
            pos += self.stride

        if self.stride > 1 and pos < L:
            tail = seq[pos:]
            if self.pad_incomplete:
                tokens.append(tail.ljust(self.k, "-"))

        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.unk_token_id
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = set()
        if skip_special_tokens:
            special_ids = {self.token2id[t] for t in self.SPECIAL_TOKENS
                           if t in self.token2id}
            special_ids |= {self.token2id[t] for t in self.organism_tokens
                            if t in self.token2id}

        real_tokens = [
            self.id2token[i]
            for i in ids
            if i in self.id2token and i not in special_ids
        ]

        if not real_tokens:
            return ""

        if self.stride == 1:
            seq = "".join(t[0] for t in real_tokens[:-1]) + real_tokens[-1]
        else:
            seq = "".join(real_tokens)

        return seq.replace("-", "")

    def _build_mutation_mask(
        self,
        tokens: list[str],
        mutation_positions: list[int] | None,
    ) -> list[int]:
        """a mutation at position p affects all k-mer windows whose span overlaps p."""
        mask = [0] * len(tokens)
        if not mutation_positions:
            return mask

        for pos in mutation_positions:
            first_tok = max(0, pos - self.k + 1)
            last_tok  = pos // self.stride
            for ti in range(first_tok, min(last_tok + 1, len(mask))):
                mask[ti] = 1

        return mask

    def __repr__(self) -> str:
        return (
            f"KmerTokenizer(k={self.k}, stride={self.stride}, "
            f"vocab_size={self.vocab_size})"
        )


class CharTokenizer(BaseProteinTokenizer):
    """
    single-character amino acid tokenizer.
    each position in the sequence maps to exactly one token, making
    this the simplest and most transparent tokenizer. ideal for:
      - single-point mutation tasks (position alignment is exact)
      - baseline models and debugging
      - any model that processes residues one-by-one (e.g. esm-style)
    """
    def __init__(
        self,
        add_special_tokens: bool = True,
        organism_tokens: list[str] | None = None,
        include_ambiguous: bool = True,
    ) -> None:
        super().__init__(
            add_special_tokens=add_special_tokens,
            organism_tokens=organism_tokens,
        )
        self.include_ambiguous = include_ambiguous
        self._build_vocab()

    def _build_vocab(self) -> None:
        tokens = (
            self.SPECIAL_TOKENS
            + self.organism_tokens
            + self.AMINO_ACIDS
            + (self.AMBIGUOUS_AA if self.include_ambiguous else [])
        )
        self._register_vocab(tokens)

    def tokenize(self, sequence: str) -> list[str]:
        return list(sequence.upper())

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.unk_token_id
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = set()
        if skip_special_tokens:
            special_ids = {self.token2id[t] for t in self.SPECIAL_TOKENS
                           if t in self.token2id}
            special_ids |= {self.token2id[t] for t in self.organism_tokens
                            if t in self.token2id}

        return "".join(
            self.id2token[i]
            for i in ids
            if i in self.id2token and i not in special_ids
        )



# helpers
def _word_to_chars(sequence: str) -> list[str]:
    return list(sequence.upper())


def _apply_merge(word: list[str], pair: tuple[str, str]) -> list[str]:
    """apply one BPE merge rule to a tokenised word in-place (returns new list)."""
    merged = "".join(pair)
    new_word: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


def _get_pair_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    stats: Counter = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += freq
    return stats


def _merge_vocab(
    vocab: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    return {
        tuple(_apply_merge(list(word), pair)): freq
        for word, freq in vocab.items()
    }


def parse_mutation_string(mut_str: str) -> ParsedMutation:
    parts = [p.strip() for p in mut_str.split(":")]
    records: list[MutationRecord] = []

    for part in parts:
        if _INSERTION_RE.match(part):
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
