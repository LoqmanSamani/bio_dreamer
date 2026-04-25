"""
K-mer tokenizer: overlapping (stride=1) or strided windows of k residues.
vocabulary layout:
  [PAD] [UNK] [CLS] [SEP] [MASK] [MUT]
  [ORG_*] ...           (optional organism tokens)
  AAA AAC AAD ...       (all k^20 possible k-mers over canonical AA)

windowing modes
---------------
  stride=1  (default, overlapping):
    "MKTAY", k=3  →  ["MKT", "KTA", "TAY"]
    length = max(0, L - k + 1)

  stride=k  (non-overlapping):
    "MKTAY", k=3  →  ["MKT", "AY?"]  (last window zero-padded if needed)

mutation mask
-------------
a mutated position at index `p` in the original sequence affects all k-mers
whose window covers position `p`, i.e. windows starting at
  max(0, p - k + 1) … min(p, L - k)
all those token positions are marked 1 in the mutation_mask.
"""
from __future__ import annotations
import itertools
from ..core.tokenizers import BaseProteinTokenizer




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
        # generate all possible k-mers (sorted for reproducibility)
        kmers = sorted(
            "".join(combo)
            for combo in itertools.product(aa, repeat=self.k)
        )
        tokens = self.SPECIAL_TOKENS + self.organism_tokens + kmers
        self._register_vocab(tokens)

   
    def tokenize(self, sequence: str) -> list[str]:
        """slide a window of size k over the sequence with the given stride"""
        seq = sequence.upper()
        L   = len(seq)
        tokens: list[str] = []

        pos = 0
        while pos + self.k <= L:
            tokens.append(seq[pos : pos + self.k])
            pos += self.stride

        # handle incomplete tail window (only when stride > 1)
        if self.stride > 1 and pos < L:
            tail = seq[pos:]
            if self.pad_incomplete:
                tail = tail.ljust(self.k, "-")   # pad with '-'
                tokens.append(tail)
            # else: drop the incomplete window
            
        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.unk_token_id
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """reconstruct the sequence from overlapping k-mer tokens"""
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
            # overlapping: first char of each token + full last token
            seq = "".join(t[0] for t in real_tokens[:-1]) + real_tokens[-1]
        else:
            seq = "".join(real_tokens)

        # strip padding characters introduced during tokenization
        return seq.replace("-", "")

    def _build_mutation_mask(
        self,
        tokens: list[str],
        mutation_positions: list[int] | None,
    ) -> list[int]:
        """
        a mutation at position p in the original sequence affects all k-mer
        windows that overlap position p.
        """
        mask = [0] * len(tokens)
        if not mutation_positions:
            return mask

        for pos in mutation_positions:
            # token indices whose window covers `pos`
            first_tok = max(0, pos - self.k + 1)
            last_tok  = pos // self.stride       # works for both stride=1 and >1
            for ti in range(first_tok, min(last_tok + 1, len(mask))):
                mask[ti] = 1

        return mask

    def __repr__(self) -> str:
        return (
            f"KmerTokenizer(k={self.k}, stride={self.stride}, "
            f"vocab_size={self.vocab_size})"
        )