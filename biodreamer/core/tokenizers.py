from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
import torch

SequenceInput = Union[str, list[str]]





class BaseProteinTokenizer(ABC):
    """
    common interface for all protein tokenizers in ProteinDreamer.
    every tokenizer must be able to:
      - encode a single sequence or a batch of sequences
      - decode token ids back to a sequence string
      - handle special tokens ([PAD], [MASK], [UNK], [CLS], [SEP], [MUT],
        and optional organism tokens)
      - return either python lists/dicts or pytorch tensors
    """
    # special token definitions (shared across all tokenizers)
    SPECIAL_TOKENS: list[str] = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[MUT]",
    ]
    # standard 20 amino acids + ambiguous characters
    AMINO_ACIDS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")   # 20 canonical
    AMBIGUOUS_AA: list[str] = list("BJOUXZ")                 # rare / ambiguous
    def __init__(
        self,
        add_special_tokens: bool = True,
        organism_tokens: list[str] | None = None,
    ) -> None:
        """
        add_special_tokens : bool
            whether [CLS] / [SEP] are prepended/appended by default.
        organism_tokens : list[str] | None
            optional list of organism/taxon tokens to add to the vocabulary,
            e.g. ["[ORG_HUMAN]", "[ORG_ECOLI]", "[ORG_YEAST]"].
        """
        self.add_special_tokens = add_special_tokens
        self.organism_tokens: list[str] = organism_tokens or []
        # built by subclass via _build_vocab()
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    # helpers
    def _register_vocab(self, tokens: list[str]) -> None:
        """assign consecutive integer ids to a list of tokens"""
        for tok in tokens:
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        return self.token2id["[PAD]"]

    @property
    def unk_token_id(self) -> int:
        return self.token2id["[UNK]"]

    @property
    def cls_token_id(self) -> int:
        return self.token2id["[CLS]"]

    @property
    def sep_token_id(self) -> int:
        return self.token2id["[SEP]"]

    @property
    def mask_token_id(self) -> int:
        return self.token2id["[MASK]"]

    @property
    def mut_token_id(self) -> int:
        return self.token2id["[MUT]"]

    @abstractmethod
    def tokenize(self, sequence: str) -> list[str]:
        """convert a raw amino acid string into a list of token strings"""

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """map token strings to integer ids"""

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """convert integer ids back to a sequence string"""

    def encode(
        self,
        sequence: str,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool | None = None,
        return_tensors: bool = False,
        mutation_positions: list[int] | None = None,
    ) -> dict:
        """encode a single amino acid sequence into token ids, attention mask, and mutation mask"""
        use_special = (
            add_special_tokens
            if add_special_tokens is not None
            else self.add_special_tokens
        )
        tokens = self.tokenize(sequence)
        ids    = self.convert_tokens_to_ids(tokens)
        # build mutation mask aligned to tokens (before special tokens)
        mut_mask = self._build_mutation_mask(tokens, mutation_positions)
        
        if use_special:
            ids      = [self.cls_token_id] + ids + [self.sep_token_id]
            mut_mask = [0] + mut_mask + [0]
            
        if truncation and max_length is not None:
            ids      = ids[:max_length]
            mut_mask = mut_mask[:max_length]
        attention_mask = [1] * len(ids)
        
        if padding and max_length is not None:
            pad_len      = max_length - len(ids)
            ids          = ids      + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            mut_mask     = mut_mask + [0] * pad_len

        if return_tensors:
            return {
                "input_ids":      torch.tensor(ids,            dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "mutation_mask":  torch.tensor(mut_mask,       dtype=torch.long),
            }
        return {
            "input_ids":      ids,
            "attention_mask": attention_mask,
            "mutation_mask":  mut_mask,
        }

    def batch_encode(
        self,
        sequences: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool | None = None,
        return_tensors: bool = False,
        mutation_positions: list[list[int]] | None = None,
    ) -> dict:
        """
        encode a batch of sequences with automatic padding to the longest
        sequence in the batch (or max_length if provided)
        """
        mut_positions_list = mutation_positions or [None] * len(sequences)

        encoded = [
            self.encode(
                seq,
                max_length=max_length,
                padding=False,      # pad after finding batch max length
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_tensors=False,
                mutation_positions=mp,
            )
            for seq, mp in zip(sequences, mut_positions_list)
        ]

        batch_max = max(len(e["input_ids"]) for e in encoded)
        if max_length is not None:
            batch_max = min(batch_max, max_length)

        if padding:
            for e in encoded:
                pad_len = batch_max - len(e["input_ids"])
                e["input_ids"]      += [self.pad_token_id] * pad_len
                e["attention_mask"] += [0]                 * pad_len
                e["mutation_mask"]  += [0]                 * pad_len

        if return_tensors:
            return {
                "input_ids":      torch.tensor(
                    [e["input_ids"]      for e in encoded], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [e["attention_mask"] for e in encoded], dtype=torch.long),
                "mutation_mask":  torch.tensor(
                    [e["mutation_mask"]  for e in encoded], dtype=torch.long),
            }
        return {
            "input_ids":      [e["input_ids"]      for e in encoded],
            "attention_mask": [e["attention_mask"] for e in encoded],
            "mutation_mask":  [e["mutation_mask"]  for e in encoded],
        }

    def _build_mutation_mask(
        self,
        tokens: list[str],
        mutation_positions: list[int] | None,
    ) -> list[int]:
        """
        build a per-token binary mask: 1 at mutated positions, 0 elsewhere.
        for character-level tokenizers position i maps directly to token i.
        K-mer tokenizers override this to handle overlapping/strided windows.
        """
        mask = [0] * len(tokens)
        if mutation_positions:
            for pos in mutation_positions:
                if 0 <= pos < len(mask):
                    mask[pos] = 1
        return mask

    # organism token helpers 
    def get_organism_token_id(self, organism: str) -> int:
        """return the token id for an organism tag"""
        tok = f"[ORG_{organism.upper()}]"
        if tok not in self.token2id:
            raise KeyError(
                f"Organism token '{tok}' not in vocabulary. "
                f"Available: {self.organism_tokens}"
            )
        return self.token2id[tok]

    def get_vocab(self) -> dict[str, int]:
        """return a copy of the full token -> id mapping"""
        return dict(self.token2id)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"special_tokens={self.SPECIAL_TOKENS + self.organism_tokens})"
        )