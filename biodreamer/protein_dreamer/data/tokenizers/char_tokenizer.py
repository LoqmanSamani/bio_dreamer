"""
character-level tokenizer: each amino acid is a single token.
vocabulary layout (in order of registration):
  [PAD] [UNK] [CLS] [SEP] [MASK] [MUT]
  [ORG_*] ...           (optional organism tokens)
  A C D E F G H I K L M N P Q R S T V W Y   (20 canonical AA)
  B J O U X Z                                (ambiguous / rare AA)
"""
from __future__ import annotations
from ..core.tokenizers import BaseProteinTokenizer





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
        """split sequence into individual characters (residues)"""
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