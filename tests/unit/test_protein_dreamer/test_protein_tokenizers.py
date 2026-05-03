from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from biodreamer.protein_dreamer.tokenizers import (
    BPETokenizer,
    CharTokenizer,
    KmerTokenizer,
    MutationRecord,
    ParsedMutation,
    ProteinTokenizer,
    _apply_merge,
    _get_pair_stats,
    _merge_vocab,
    apply_mutations,
    parse_mutation_string,
)






class TestMutationRecord:
    def test_position_0(self):
        rec = MutationRecord(wt_aa="A", position=42, mut_aa="G")
        assert rec.position_0 == 41

    def test_position_0_first_residue(self):
        rec = MutationRecord(wt_aa="A", position=1, mut_aa="G")
        assert rec.position_0 == 0

    def test_is_synonymous_true(self):
        rec = MutationRecord(wt_aa="A", position=1, mut_aa="A")
        assert rec.is_synonymous

    def test_is_synonymous_false(self):
        rec = MutationRecord(wt_aa="A", position=1, mut_aa="G")
        assert not rec.is_synonymous

    def test_is_deletion_true(self):
        rec = MutationRecord(wt_aa="A", position=1, mut_aa="-")
        assert rec.is_deletion

    def test_is_deletion_false(self):
        rec = MutationRecord(wt_aa="A", position=1, mut_aa="G")
        assert not rec.is_deletion

    def test_str_uses_raw_when_set(self):
        rec = MutationRecord(wt_aa="A", position=42, mut_aa="G", raw="A42G")
        assert str(rec) == "A42G"

    def test_str_formats_when_no_raw(self):
        rec = MutationRecord(wt_aa="A", position=42, mut_aa="G")
        assert str(rec) == "A42G"





class TestParsedMutation:
    def _single(self) -> ParsedMutation:
        return ParsedMutation(
            records=[MutationRecord("A", 42, "G", "A42G")],
            raw_string="A42G",
        )

    def _multi(self) -> ParsedMutation:
        return ParsedMutation(
            records=[
                MutationRecord("A", 42, "G", "A42G"),
                MutationRecord("L", 100, "V", "L100V"),
            ],
            raw_string="A42G:L100V",
        )

    def test_n_mutations_single(self):
        assert self._single().n_mutations == 1

    def test_n_mutations_multi(self):
        assert self._multi().n_mutations == 2

    def test_is_single_true(self):
        assert self._single().is_single

    def test_is_single_false(self):
        assert not self._multi().is_single

    def test_has_indels_false(self):
        assert not self._single().has_indels

    def test_has_indels_true(self):
        pm = ParsedMutation(
            records=[MutationRecord("A", 5, "-", "A5-")],
            raw_string="A5-",
        )
        assert pm.has_indels

    def test_positions_0_single(self):
        assert self._single().positions_0 == [41]

    def test_positions_0_multi(self):
        assert self._multi().positions_0 == [41, 99]






class TestParseMutationString:
    def test_single_substitution(self):
        pm = parse_mutation_string("A42G")
        assert pm.n_mutations == 1
        rec = pm.records[0]
        assert rec.wt_aa == "A"
        assert rec.position == 42
        assert rec.mut_aa == "G"

    def test_multi_substitution(self):
        pm = parse_mutation_string("A42G:L100V")
        assert pm.n_mutations == 2
        assert pm.records[1].wt_aa == "L"
        assert pm.records[1].position == 100

    def test_raw_string_preserved(self):
        pm = parse_mutation_string("A42G:L100V")
        assert pm.raw_string == "A42G:L100V"

    def test_uppercase_normalised(self):
        pm = parse_mutation_string("a42g")
        assert pm.records[0].wt_aa == "A"
        assert pm.records[0].mut_aa == "G"

    def test_deletion_parsed(self):
        pm = parse_mutation_string("A42-")
        assert pm.records[0].is_deletion

    def test_insertion_flagged(self):
        pm = parse_mutation_string("42ins3")
        assert pm.records[0].mut_aa == "ins"
        assert pm.records[0].position == 0

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse mutation token"):
            parse_mutation_string("INVALID")

    def test_stop_codon_accepted(self):
        pm = parse_mutation_string("A42*")
        assert pm.records[0].mut_aa == "*"






class TestApplyMutations:
    _SEQ = "ACDEFGHIKLM"

    def test_substitution(self):
        pm = parse_mutation_string("A1G")
        result = apply_mutations(self._SEQ, pm)
        assert result[0] == "G"
        assert result[1:] == self._SEQ[1:]

    def test_deletion_removes_residue(self):
        pm = parse_mutation_string("A1-")
        result = apply_mutations(self._SEQ, pm)
        assert len(result) == len(self._SEQ) - 1
        assert result[0] == "C"

    def test_wt_mismatch_raises(self):
        pm = parse_mutation_string("G1V")  # wt G does not match actual 'A'
        with pytest.raises(ValueError, match="WT mismatch"):
            apply_mutations(self._SEQ, pm)

    def test_out_of_range_raises(self):
        pm = parse_mutation_string("A999G")
        with pytest.raises(IndexError, match="out of range"):
            apply_mutations(self._SEQ, pm)

    def test_multi_mutation(self):
        pm = parse_mutation_string("A1G:C2D")
        result = apply_mutations(self._SEQ, pm)
        assert result[:2] == "GD"

    def test_insertion_skipped(self):
        pm = parse_mutation_string("42ins3")
        result = apply_mutations(self._SEQ, pm)
        assert result == self._SEQ.upper()  # unchanged






class TestCharTokenizer:
    @pytest.fixture
    def tok(self) -> CharTokenizer:
        return CharTokenizer()

    def test_vocab_contains_all_amino_acids(self, tok):
        vocab = tok.get_vocab()
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert aa in vocab

    def test_vocab_contains_special_tokens(self, tok):
        for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[MUT]"):
            assert t in tok.get_vocab()

    def test_vocab_size_canonical(self):
        tok = CharTokenizer(include_ambiguous=False)
        # 6 special + 20 canonical
        assert tok.vocab_size == 26

    def test_vocab_size_with_ambiguous(self, tok):
        # 6 special + 20 canonical + 6 ambiguous
        assert tok.vocab_size == 32

    def test_encode_output_keys(self, tok):
        out = tok.encode("AC")
        assert set(out.keys()) == {"input_ids", "attention_mask", "mutation_mask"}

    def test_encode_length_with_special_tokens(self, tok):
        out = tok.encode("ACDEF")
        # [CLS] + 5 tokens + [SEP]
        assert len(out["input_ids"]) == 7

    def test_encode_no_special_tokens(self):
        tok = CharTokenizer(add_special_tokens=False)
        out = tok.encode("ACDEF")
        assert len(out["input_ids"]) == 5

    def test_encode_cls_sep_positions(self, tok):
        out = tok.encode("AC")
        assert out["input_ids"][0] == tok.cls_token_id
        assert out["input_ids"][-1] == tok.sep_token_id

    def test_attention_mask_all_ones(self, tok):
        out = tok.encode("ACDEF")
        assert all(v == 1 for v in out["attention_mask"])

    def test_mutation_mask_marks_position(self, tok):
        out = tok.encode("ACDEF", mutation_positions=[1])
        # position 1 → token index 2 (after [CLS])
        assert out["mutation_mask"][2] == 1
        assert out["mutation_mask"][0] == 0  # [CLS] never marked

    def test_mutation_mask_no_positions(self, tok):
        out = tok.encode("ACDEF")
        assert all(v == 0 for v in out["mutation_mask"])

    def test_encode_return_tensors(self, tok):
        out = tok.encode("ACDEF", return_tensors=True)
        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].dtype == torch.long

    def test_encode_truncation(self, tok):
        out = tok.encode("ACDEFGHIK", max_length=5, truncation=True)
        assert len(out["input_ids"]) == 5

    def test_encode_padding(self, tok):
        out = tok.encode("AC", max_length=10, padding=True)
        assert len(out["input_ids"]) == 10
        assert out["input_ids"][-1] == tok.pad_token_id
        assert out["attention_mask"][-1] == 0

    def test_decode_roundtrip(self, tok):
        seq = "ACDEFGHIK"
        ids = tok.encode(seq)["input_ids"]
        assert tok.decode(ids) == seq

    def test_decode_skip_special_false(self, tok):
        ids = tok.encode("AC")["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert "[CLS]" in decoded

    def test_unknown_token_maps_to_unk(self, tok):
        out = tok.encode("1")  # '1' is not an amino acid
        assert out["input_ids"][1] == tok.unk_token_id

    def test_batch_encode_pads_to_longest(self, tok):
        out = tok.batch_encode(["AC", "ACDEF"])
        lengths = [len(row) for row in out["input_ids"]]
        assert lengths[0] == lengths[1]

    def test_batch_encode_return_tensors_shape(self, tok):
        out = tok.batch_encode(["AC", "ACDEF"], return_tensors=True)
        assert out["input_ids"].shape == (2, out["input_ids"].shape[1])

    def test_organism_token(self):
        tok = CharTokenizer(organism_tokens=["[ORG_HUMAN]"])
        assert tok.get_organism_token_id("human") == tok.token2id["[ORG_HUMAN]"]

    def test_unknown_organism_raises(self, tok):
        with pytest.raises(KeyError):
            tok.get_organism_token_id("human")






class TestKmerTokenizer:
    @pytest.fixture
    def tok3(self) -> KmerTokenizer:
        return KmerTokenizer(k=3, stride=1)

    def test_k1_same_as_char(self):
        tok = KmerTokenizer(k=1)
        tokens = tok.tokenize("ACDEF")
        assert tokens == ["A", "C", "D", "E", "F"]

    def test_tokenize_k3_stride1_count(self, tok3):
        # 5-residue seq → 3 overlapping 3-mers
        tokens = tok3.tokenize("ACDEF")
        assert len(tokens) == 3

    def test_tokenize_k3_stride3_count(self):
        tok = KmerTokenizer(k=3, stride=3)
        tokens = tok.tokenize("ACDEFGHI")
        # windows at 0 and 3 → 2 full k-mers, tail dropped
        assert len(tokens) == 2

    def test_tokenize_k3_stride3_pad_incomplete(self):
        tok = KmerTokenizer(k=3, stride=3, pad_incomplete=True)
        tokens = tok.tokenize("ACDEFG")
        # "ACD", "EFG" → no tail
        tokens2 = tok.tokenize("ACDEFGH")
        # tail "H" padded → "H--"
        assert len(tokens2) == len(tokens) + 1

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            KmerTokenizer(k=0)

    def test_encode_length_k3_stride1(self, tok3):
        out = tok3.encode("ACDEF")
        # [CLS] + 3 k-mers + [SEP]
        assert len(out["input_ids"]) == 5

    def test_mutation_mask_k3_stride1_coverage(self, tok3):
        # mutation at position 2 in "ACDEF" → tokens covering char 2:
        # token 0 = ACD covers [0,2], token 1 = CDE covers [1,3], token 2 = DEF covers [2,4]
        # with [CLS] offset, token indices in output are 1,2,3
        out = tok3.encode("ACDEF", mutation_positions=[2])
        mut_mask = out["mutation_mask"]
        assert mut_mask[1] == 1  # token 0 + [CLS] offset
        assert mut_mask[2] == 1
        assert mut_mask[3] == 1

    def test_mutation_mask_zero_for_cls_sep(self, tok3):
        out = tok3.encode("ACDEF", mutation_positions=[0])
        assert out["mutation_mask"][0] == 0  # [CLS]
        assert out["mutation_mask"][-1] == 0  # [SEP]

    def test_decode_stride1_roundtrip(self, tok3):
        seq = "ACDEFGHIK"
        ids = tok3.encode(seq)["input_ids"]
        assert tok3.decode(ids) == seq

    def test_vocab_contains_all_3mers(self, tok3):
        vocab = tok3.get_vocab()
        # ACA is a valid 3-mer
        assert "ACA" in vocab

    def test_batch_encode_pads(self, tok3):
        out = tok3.batch_encode(["ACG", "ACGDEF"])
        lengths = [len(row) for row in out["input_ids"]]
        assert lengths[0] == lengths[1]






class TestBPETokenizer:
    _SEQS = ["ACDEF"] * 10 + ["ACDAC"] * 10

    @pytest.fixture
    def untrained(self) -> BPETokenizer:
        return BPETokenizer(vocab_size=50)

    @pytest.fixture
    def trained(self) -> BPETokenizer:
        tok = BPETokenizer(vocab_size=50)
        tok.train(self._SEQS, min_frequency=2)
        return tok

    def test_seed_vocab_contains_amino_acids(self, untrained):
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert aa in untrained.get_vocab()

    def test_seed_vocab_contains_special_tokens(self, untrained):
        for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[MUT]"):
            assert t in untrained.get_vocab()

    def test_tokenize_without_merges_is_chars(self, untrained):
        assert untrained.tokenize("ACDEF") == ["A", "C", "D", "E", "F"]

    def test_train_increases_vocab(self, untrained):
        size_before = untrained.vocab_size
        untrained.train(self._SEQS, min_frequency=2)
        assert untrained.vocab_size > size_before

    def test_train_adds_merges(self, trained):
        assert len(trained.merges) > 0

    def test_tokenize_after_training_merges_pairs(self, trained):
        tokens = trained.tokenize("ACDEF")
        # after training on ACDEF-heavy corpus, ACD should be merged into fewer tokens
        assert len(tokens) < len("ACDEF")

    def test_train_respects_target_vocab_size(self):
        tok = BPETokenizer(vocab_size=40)
        tok.train(self._SEQS * 5, min_frequency=1)
        assert tok.vocab_size <= 40

    def test_inject_merges_at_init(self):
        tok = BPETokenizer(vocab_size=100, merges=[("A", "C")])
        assert ("A", "C") in tok.merges
        assert "AC" in tok.get_vocab()

    def test_encode_output_shape(self, trained):
        out = trained.encode("ACDEF")
        assert len(out["input_ids"]) >= 2  # at least [CLS] + [SEP]

    def test_decode_roundtrip(self, untrained):
        # without merges char-level roundtrip is exact
        seq = "ACDEF"
        ids = untrained.encode(seq)["input_ids"]
        assert untrained.decode(ids) == seq

    def test_mutation_mask_span(self, trained):
        # mutation at position 0; the token covering char 0 must be marked
        out = trained.encode("ACDEF", mutation_positions=[0])
        # [CLS] is index 0; first real token is index 1
        assert out["mutation_mask"][1] == 1

    def test_save_load_roundtrip(self, trained):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bpe.json"
            trained.save(path)
            loaded = BPETokenizer.load(path)
        assert loaded.merges == trained.merges
        assert loaded.target_vocab_size == trained.target_vocab_size

    def test_save_roundtrip_tokenizes_same(self, trained):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bpe.json"
            trained.save(path)
            loaded = BPETokenizer.load(path)
        assert loaded.tokenize("ACDEF") == trained.tokenize("ACDEF")

    def test_load_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"type": "CharTokenizer"}))
            with pytest.raises(ValueError, match="BPETokenizer"):
                BPETokenizer.load(path)

    def test_repr_untrained(self, untrained):
        assert "untrained" in repr(untrained)

    def test_repr_trained(self, trained):
        assert "merges" in repr(trained)






class TestProteinTokenizer:
    def test_repr_class_name(self):
        tok = ProteinTokenizer(mode="char")
        assert repr(tok).startswith("ProteinTokenizer(")

    def test_repr_contains_mode(self):
        tok = ProteinTokenizer(mode="kmer")
        assert "mode='kmer'" in repr(tok)

    def test_mode_char_vocab_size(self):
        tok = ProteinTokenizer(mode="char")
        assert tok.vocab_size == 32  # 6 special + 20 + 6 ambiguous

    def test_mode_kmer_created(self):
        tok = ProteinTokenizer(mode="kmer", k=2)
        out = tok.encode_sequence("ACDEF")
        assert len(out["input_ids"]) > 0

    def test_mode_bpe_created(self):
        tok = ProteinTokenizer(mode="bpe", bpe_vocab_size=50)
        out = tok.encode_sequence("ACDEF")
        assert len(out["input_ids"]) > 0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown tokenizer mode"):
            ProteinTokenizer(mode="invalid")

    def test_pad_token_id_accessible(self):
        tok = ProteinTokenizer(mode="char")
        assert isinstance(tok.pad_token_id, int)

    def test_mask_token_id_accessible(self):
        tok = ProteinTokenizer(mode="char")
        assert isinstance(tok.mask_token_id, int)

    def test_get_vocab_returns_dict(self):
        tok = ProteinTokenizer(mode="char")
        vocab = tok.get_vocab()
        assert isinstance(vocab, dict)

    def test_decode_roundtrip(self):
        tok = ProteinTokenizer(mode="char")
        seq = "ACDEFGHIK"
        ids = tok.encode_sequence(seq)["input_ids"]
        assert tok.decode(ids) == seq

    def test_batch_encode_sequences_shape(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.batch_encode_sequences(["AC", "ACDEF"], return_tensors=True)
        assert out["input_ids"].shape[0] == 2

    # encode_mutation_string
    def test_encode_mutation_string_substitution(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.encode_mutation_string("ACDEF", "A1G")
        assert "parsed_mutation" in out
        assert out["parsed_mutation"].records[0].mut_aa == "G"

    def test_encode_mutation_string_marks_position(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.encode_mutation_string("ACDEF", "A1G")
        # position 0 → token index 1 (after [CLS])
        assert out["mutation_mask"][1] == 1

    def test_encode_mutation_strict_false_fallback(self):
        tok = ProteinTokenizer(mode="char")
        # G1G has wt=G but actual is A — should fall back silently
        out = tok.encode_mutation_string("ACDEF", "G1V", strict_wt_check=False)
        assert len(out["input_ids"]) > 0

    def test_encode_mutation_strict_true_raises(self):
        tok = ProteinTokenizer(mode="char")
        with pytest.raises(ValueError):
            tok.encode_mutation_string("ACDEF", "G1V", strict_wt_check=True)

    # batch_encode_mutations
    def test_batch_encode_mutations_lengths_equal(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.batch_encode_mutations(
            ["ACDEF", "ACDEFGHIK"],
            ["A1G",   "A1G"],
            padding=True,
        )
        lengths = [len(r) for r in out["input_ids"]]
        assert lengths[0] == lengths[1]

    def test_batch_encode_mutations_return_tensors(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.batch_encode_mutations(
            ["ACDEF", "ACDEF"],
            ["A1G", "C2D"],
            return_tensors=True,
        )
        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].shape[0] == 2

    def test_batch_encode_mutations_parsed_list(self):
        tok = ProteinTokenizer(mode="char")
        out = tok.batch_encode_mutations(["ACDEF"], ["A1G"])
        assert len(out["parsed_mutations"]) == 1

    # encode_with_organism
    def test_encode_with_organism_prepends_token(self):
        tok = ProteinTokenizer(mode="char", organism_tokens=["[ORG_HUMAN]"])
        out = tok.encode_with_organism("ACDEF", "human")
        vocab = tok.get_vocab()
        assert out["input_ids"][0] == vocab["[ORG_HUMAN]"]

    def test_encode_with_organism_unknown_raises(self):
        tok = ProteinTokenizer(mode="char")
        with pytest.raises(KeyError):
            tok.encode_with_organism("ACDEF", "human")

    def test_encode_with_organism_padding(self):
        tok = ProteinTokenizer(mode="char", organism_tokens=["[ORG_HUMAN]"])
        out = tok.encode_with_organism("ACDEF", "human", max_length=20, padding=True)
        assert len(out["input_ids"]) == 20

    # BPE-specific helpers via wrapper
    def test_train_bpe_on_non_bpe_raises(self):
        tok = ProteinTokenizer(mode="char")
        with pytest.raises(RuntimeError, match="mode='bpe'"):
            tok.train_bpe(["ACDEF"])

    def test_save_bpe_on_non_bpe_raises(self):
        tok = ProteinTokenizer(mode="char")
        with pytest.raises(RuntimeError, match="mode='bpe'"):
            tok.save_bpe("/tmp/bpe.json")

    def test_train_and_save_bpe(self):
        tok = ProteinTokenizer(mode="bpe", bpe_vocab_size=50)
        tok.train_bpe(["ACDEF"] * 20, min_frequency=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bpe.json")
            tok.save_bpe(path)
            loaded = ProteinTokenizer.load_bpe(path)
        assert loaded.vocab_size == tok.vocab_size






class TestApplyMerge:
    def test_basic_merge(self):
        assert _apply_merge(["A", "C", "D"], ("A", "C")) == ["AC", "D"]

    def test_no_match_unchanged(self):
        assert _apply_merge(["A", "C", "D"], ("X", "Y")) == ["A", "C", "D"]

    def test_multiple_occurrences(self):
        result = _apply_merge(["A", "C", "A", "C"], ("A", "C"))
        assert result == ["AC", "AC"]

    def test_non_overlapping(self):
        # "AC" and "CA" both present, only "AC" merged
        result = _apply_merge(["A", "C", "A"], ("A", "C"))
        assert result == ["AC", "A"]

    def test_single_element_list(self):
        assert _apply_merge(["A"], ("A", "C")) == ["A"]

    def test_empty_list(self):
        assert _apply_merge([], ("A", "C")) == []






class TestMergeVocab:
    def test_applies_merge_to_all_words(self):
        vocab = {("A", "C", "D"): 3, ("A", "C", "E"): 2}
        result = _merge_vocab(vocab, ("A", "C"))
        assert ("AC", "D") in result
        assert ("AC", "E") in result

    def test_frequencies_preserved(self):
        vocab = {("A", "C", "D"): 5}
        result = _merge_vocab(vocab, ("A", "C"))
        assert result[("AC", "D")] == 5

    def test_no_match_unchanged(self):
        vocab = {("A", "C", "D"): 2}
        result = _merge_vocab(vocab, ("X", "Y"))
        assert result == vocab






class TestGetPairStats:
    def test_counts_adjacent_pairs(self):
        vocab = {("A", "C", "D"): 2}
        stats = _get_pair_stats(vocab)
        assert stats[("A", "C")] == 2
        assert stats[("C", "D")] == 2

    def test_accumulates_across_words(self):
        vocab = {("A", "C"): 3, ("B", "C"): 2}
        stats = _get_pair_stats(vocab)
        assert stats[("A", "C")] == 3
        assert stats[("B", "C")] == 2

    def test_single_char_word_no_pairs(self):
        vocab = {("A",): 10}
        stats = _get_pair_stats(vocab)
        assert len(stats) == 0
