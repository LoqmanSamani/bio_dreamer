"""
byte-pair encoding (bpe) tokenizer for amino acid sequences.

unlike char/k-mer tokenizers whose vocabularies are fixed by construction,
bpe learns a merge table from a corpus of sequences.  frequent pairs of
tokens are iteratively merged until the target vocabulary size is reached.

vocabulary layout (after training):
  [PAD] [UNK] [CLS] [SEP] [MASK] [MUT]
  [ORG_*] ...          (optional organism tokens)
  A C D E F ...        (20 canonical + optional ambiguous, seed vocabulary)
  AC CD DE ...         (learned merges, most frequent first)

workflow
--------
1.  train on a list of sequences:
        tok = BPETokenizer(vocab_size=500)
        tok.train(sequences)                    # learns merge table

2.  save / load:
        tok.save("config/protein_dreamer/bpe_vocab.json")
        tok2 = BPETokenizer.load("config/protein_dreamer/bpe_vocab.json")

3.  encode:
        out = tok.encode("MKTAY...", return_tensors=True)

bpe and mutation positions
--------------------------
bpe tokens are variable-length, so a mutation at position p may fall
inside a merged token that also covers unmutated residues. the mutation
mask marks every token whose character span overlaps position p.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator
from ..core.tokenizers import BaseProteinTokenizer





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
        # ordered list of (a, b) → "ab" merge rules
        self.merges: list[tuple[str, str]] = merges or []
        # build seed vocab (specials + single-char aa tokens)
        self._build_seed_vocab()
        # if merges were injected, apply them to the vocab immediately
        if self.merges:
            self._apply_merges_to_vocab(self.merges)

    def _build_seed_vocab(self) -> None:
        """register specials + individual amino acid characters"""
        seed = (
            self.SPECIAL_TOKENS
            + self.organism_tokens
            + self.AMINO_ACIDS
            + (self.AMBIGUOUS_AA if self.include_ambiguous else [])
        )
        self._register_vocab(seed)

    def _apply_merges_to_vocab(self, merges: list[tuple[str, str]]) -> None:
        """register merged tokens derived from the merge table"""
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
        """learn bpe merge rules from a list of amino acid sequences"""
        working_vocab: dict[tuple[str, ...], int] = Counter(
            tuple(_word_to_chars(seq))
            for seq in sequences
            if seq.strip()
        )
        n_seed      = self.vocab_size # current vocab size (seed already registered)
        n_special   = len(self.SPECIAL_TOKENS) + len(self.organism_tokens)
        max_merges  = self.target_vocab_size - n_seed
        
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
            # filter by minimum frequency
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
        """apply learned bpe merges to a sequence and return token strings"""
        if not self.merges:
            return _word_to_chars(sequence)
        word = _word_to_chars(sequence.upper())
        for pair in self.merges:
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
            word = new_word
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
        """bpe tokens are variable-length, so we must track character spans.
        token i covers characters [span_start, span_start + len(token) - 1].
        a mutation at position p marks every token whose span includes p.
        """
        mask = [0] * len(tokens)
        if not mutation_positions:
            return mask
        # build character-offset spans for each token
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
        """serialise the tokenizer to a json file"""
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
        """load a previously saved BPETokenizer from a json file"""
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
        
        
        
# helpers 
def _word_to_chars(sequence: str) -> list[str]:
    """split a sequence into individual characters (seed bpe units)"""
    return list(sequence.upper())

def _get_pair_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    """count adjacent pair frequencies across all words in the vocab"""
    stats: Counter = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += freq
    return stats

def _merge_vocab(
    vocab: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """apply one bpe merge to every word in the working vocabulary"""
    merged   = "".join(pair)
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab