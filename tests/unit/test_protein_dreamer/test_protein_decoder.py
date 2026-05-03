from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from biodreamer.core.decoder import BaseDecoder
from biodreamer.protein_dreamer.decoder import (
    ProteinSequenceDecoder,
    ProteinStructureDecoder,
)

LATENT_DIM = 32
BATCH = 4
SEQ_LEN = 10
DEFAULT_LEN = 8
MAX_LEN = 64
N_LAYERS = 2
N_HEADS = 4





@pytest.fixture
def seq_dec() -> ProteinSequenceDecoder:
    return ProteinSequenceDecoder(
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_LEN,
        default_seq_len=DEFAULT_LEN,
        device=torch.device("cpu"),
    )


@pytest.fixture
def struct_dec() -> ProteinStructureDecoder:
    return ProteinStructureDecoder(
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_LEN,
        default_seq_len=DEFAULT_LEN,
        device=torch.device("cpu"),
    )


@pytest.fixture
def z() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, LATENT_DIM)




class TestSeqDecoderConstruction:
    def test_is_base_decoder(self, seq_dec):
        assert isinstance(seq_dec, BaseDecoder)

    def test_is_nn_module(self, seq_dec):
        assert isinstance(seq_dec, nn.Module)

    def test_latent_dim_stored(self, seq_dec):
        assert seq_dec.latent_dim == LATENT_DIM

    def test_get_latent_dim(self, seq_dec):
        assert seq_dec.get_latent_dim() == LATENT_DIM

    def test_default_seq_len_stored(self, seq_dec):
        assert seq_dec.default_seq_len == DEFAULT_LEN

    def test_max_seq_len_stored(self, seq_dec):
        assert seq_dec.max_seq_len == MAX_LEN

    def test_device_is_cpu(self, seq_dec):
        assert seq_dec.device == torch.device("cpu")

    def test_pos_queries_shape(self, seq_dec):
        assert seq_dec.pos_queries.shape == (MAX_LEN, LATENT_DIM)

    def test_pos_queries_is_parameter(self, seq_dec):
        assert isinstance(seq_dec.pos_queries, nn.Parameter)

    def test_logit_head_output_dim(self, seq_dec):
        assert seq_dec.logit_head.out_features == 20

    def test_has_layers(self, seq_dec):
        assert len(seq_dec.layers) == N_LAYERS

    def test_has_norm(self, seq_dec):
        assert isinstance(seq_dec.norm, nn.LayerNorm)

    def test_has_z_proj(self, seq_dec):
        assert isinstance(seq_dec.z_proj, nn.Linear)
        assert seq_dec.z_proj.in_features == LATENT_DIM
        assert seq_dec.z_proj.out_features == LATENT_DIM




class TestSeqDecoderDecode:
    def test_output_keys(self, seq_dec, z):
        out = seq_dec.decode(z)
        assert set(out.keys()) == {"logits", "sequences"}

    def test_logits_shape_default_len(self, seq_dec, z):
        logits = seq_dec.decode(z)["logits"]
        assert logits.shape == (BATCH, DEFAULT_LEN, 20)

    def test_logits_shape_explicit_len(self, seq_dec, z):
        logits = seq_dec.decode(z, seq_len=SEQ_LEN)["logits"]
        assert logits.shape == (BATCH, SEQ_LEN, 20)

    def test_sequences_count(self, seq_dec, z):
        sequences = seq_dec.decode(z)["sequences"]
        assert len(sequences) == BATCH

    def test_sequences_length(self, seq_dec, z):
        sequences = seq_dec.decode(z, seq_len=SEQ_LEN)["sequences"]
        for seq in sequences:
            assert len(seq) == SEQ_LEN

    def test_sequences_are_strings(self, seq_dec, z):
        for seq in seq_dec.decode(z)["sequences"]:
            assert isinstance(seq, str)

    def test_sequences_contain_valid_aa(self, seq_dec, z):
        from biodreamer.protein_dreamer.utils import _AA_LIST
        valid = set(_AA_LIST)
        for seq in seq_dec.decode(z)["sequences"]:
            assert all(c in valid for c in seq)

    def test_logits_finite(self, seq_dec, z):
        assert torch.isfinite(seq_dec.decode(z)["logits"]).all()

    def test_logits_dtype_float(self, seq_dec, z):
        logits = seq_dec.decode(z)["logits"]
        assert logits.dtype == torch.float32

    def test_seq_len_1_works(self, seq_dec, z):
        out = seq_dec.decode(z, seq_len=1)
        assert out["logits"].shape == (BATCH, 1, 20)
        assert all(len(s) == 1 for s in out["sequences"])

    def test_seq_len_exceeds_max_raises(self, seq_dec, z):
        with pytest.raises(ValueError, match="max_seq_len"):
            seq_dec.decode(z, seq_len=MAX_LEN + 1)

    def test_batch_size_1(self, seq_dec):
        z = torch.randn(1, LATENT_DIM)
        out = seq_dec.decode(z)
        assert out["logits"].shape == (1, DEFAULT_LEN, 20)

    def test_gradient_flows_through_logits(self, seq_dec):
        z = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
        logits = seq_dec.decode(z)["logits"]
        logits.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_different_z_different_logits(self, seq_dec):
        z1 = torch.randn(BATCH, LATENT_DIM)
        z2 = torch.randn(BATCH, LATENT_DIM)
        out1 = seq_dec.decode(z1)["logits"]
        out2 = seq_dec.decode(z2)["logits"]
        assert not torch.allclose(out1, out2)

    def test_same_z_same_logits_in_eval(self, seq_dec, z):
        seq_dec.eval()
        with torch.no_grad():
            out1 = seq_dec.decode(z)["logits"]
            out2 = seq_dec.decode(z)["logits"]
        assert torch.allclose(out1, out2)




class TestSeqDecoderSetSeqLen:
    def test_updates_default(self, seq_dec, z):
        seq_dec.set_seq_len(SEQ_LEN)
        assert seq_dec.default_seq_len == SEQ_LEN

    def test_decode_uses_new_default(self, seq_dec, z):
        seq_dec.set_seq_len(SEQ_LEN)
        logits = seq_dec.decode(z)["logits"]
        assert logits.shape == (BATCH, SEQ_LEN, 20)

    def test_forward_delegates_to_decode(self, seq_dec, z):
        seq_dec.eval()
        with torch.no_grad():
            out_fwd = seq_dec.forward(z, seq_len=SEQ_LEN)
            out_dec = seq_dec.decode(z, seq_len=SEQ_LEN)
        assert torch.allclose(out_fwd["logits"], out_dec["logits"])

    def test_nn_module_forward_works(self, seq_dec, z):
        out = seq_dec(z, seq_len=SEQ_LEN)
        assert out["logits"].shape == (BATCH, SEQ_LEN, 20)

    def test_decode_batch_equals_decode(self, seq_dec, z):
        seq_dec.eval()
        with torch.no_grad():
            out_db = seq_dec.decode_batch(z)
            out_d = seq_dec.decode(z)
        assert torch.allclose(out_db["logits"], out_d["logits"])




class TestStructDecoderConstruction:
    def test_is_base_decoder(self, struct_dec):
        assert isinstance(struct_dec, BaseDecoder)

    def test_is_nn_module(self, struct_dec):
        assert isinstance(struct_dec, nn.Module)

    def test_latent_dim_stored(self, struct_dec):
        assert struct_dec.latent_dim == LATENT_DIM

    def test_get_latent_dim(self, struct_dec):
        assert struct_dec.get_latent_dim() == LATENT_DIM

    def test_coord_head_output_dim(self, struct_dec):
        assert struct_dec.coord_head.out_features == 3

    def test_pos_queries_shape(self, struct_dec):
        assert struct_dec.pos_queries.shape == (MAX_LEN, LATENT_DIM)

    def test_has_layers(self, struct_dec):
        assert len(struct_dec.layers) == N_LAYERS

    def test_device_is_cpu(self, struct_dec):
        assert struct_dec.device == torch.device("cpu")




class TestStructDecoderDecode:
    def test_output_keys(self, struct_dec, z):
        out = struct_dec.decode(z)
        assert set(out.keys()) == {"coords"}

    def test_coords_shape_default_len(self, struct_dec, z):
        coords = struct_dec.decode(z)["coords"]
        assert coords.shape == (BATCH, DEFAULT_LEN, 3)

    def test_coords_shape_explicit_len(self, struct_dec, z):
        coords = struct_dec.decode(z, seq_len=SEQ_LEN)["coords"]
        assert coords.shape == (BATCH, SEQ_LEN, 3)

    def test_coords_finite(self, struct_dec, z):
        assert torch.isfinite(struct_dec.decode(z)["coords"]).all()

    def test_coords_dtype_float(self, struct_dec, z):
        coords = struct_dec.decode(z)["coords"]
        assert coords.dtype == torch.float32

    def test_batch_size_1(self, struct_dec):
        z = torch.randn(1, LATENT_DIM)
        coords = struct_dec.decode(z, seq_len=SEQ_LEN)["coords"]
        assert coords.shape == (1, SEQ_LEN, 3)

    def test_gradient_flows_through_coords(self, struct_dec):
        z = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
        coords = struct_dec.decode(z)["coords"]
        coords.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_different_z_different_coords(self, struct_dec):
        z1 = torch.randn(BATCH, LATENT_DIM)
        z2 = torch.randn(BATCH, LATENT_DIM)
        c1 = struct_dec.decode(z1)["coords"]
        c2 = struct_dec.decode(z2)["coords"]
        assert not torch.allclose(c1, c2)

    def test_seq_len_exceeds_max_raises(self, struct_dec, z):
        with pytest.raises(ValueError, match="max_seq_len"):
            struct_dec.decode(z, seq_len=MAX_LEN + 1)

    def test_set_seq_len_updates_default(self, struct_dec, z):
        struct_dec.set_seq_len(SEQ_LEN)
        coords = struct_dec.decode(z)["coords"]
        assert coords.shape == (BATCH, SEQ_LEN, 3)

    def test_forward_delegates_to_decode(self, struct_dec, z):
        struct_dec.eval()
        with torch.no_grad():
            out_fwd = struct_dec.forward(z, seq_len=SEQ_LEN)
            out_dec = struct_dec.decode(z, seq_len=SEQ_LEN)
        assert torch.allclose(out_fwd["coords"], out_dec["coords"])

    def test_nn_module_forward_works(self, struct_dec, z):
        out = struct_dec(z, seq_len=SEQ_LEN)
        assert out["coords"].shape == (BATCH, SEQ_LEN, 3)

    def test_seq_len_1_works(self, struct_dec, z):
        coords = struct_dec.decode(z, seq_len=1)["coords"]
        assert coords.shape == (BATCH, 1, 3)




class TestDecoderConsistency:
    def test_both_use_same_n_layers(self, seq_dec, struct_dec):
        assert len(seq_dec.layers) == len(struct_dec.layers)

    def test_same_z_different_output_types(self, seq_dec, struct_dec, z):
        seq_out = seq_dec.decode(z)
        struct_out = struct_dec.decode(z)
        assert "logits" in seq_out
        assert "coords" in struct_out

    def test_seq_dec_no_coords_key(self, seq_dec, z):
        assert "coords" not in seq_dec.decode(z)

    def test_struct_dec_no_logits_key(self, struct_dec, z):
        assert "logits" not in struct_dec.decode(z)
