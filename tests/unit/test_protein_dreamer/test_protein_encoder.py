"""Unit tests for biodreamer.protein_dreamer.encoder."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from biodreamer.protein_dreamer.encoder import ActionEncoder, ProteinEncoder
from biodreamer.protein_dreamer.data.preprocessing import build_protein_graph

LATENT_DIM = 64
EMBED_DIM = 32
SEQ_LEN = 20
SEQ_HIDDEN = 128  # stub ESM-2 hidden size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_esm2(hidden_size: int = SEQ_HIDDEN) -> MagicMock:
    """Minimal ESM-2 stand-in that returns a last_hidden_state tensor."""
    model = MagicMock(spec=nn.Module)
    model.config = MagicMock()
    model.config.hidden_size = hidden_size
    model.parameters = MagicMock(return_value=iter([]))
    model.to = MagicMock(return_value=model)

    def _forward(**kwargs):
        # determine batch+seq length from input_ids
        input_ids = kwargs.get("input_ids", torch.zeros(1, SEQ_LEN + 2, dtype=torch.long))
        B, L = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(B, L, hidden_size)
        return out

    model.__call__ = MagicMock(side_effect=_forward)
    return model


def _mock_tokenizer() -> MagicMock:
    """Stand-in tokenizer that returns input_ids + attention_mask tensors."""
    tok = MagicMock()

    def _call(seqs, return_tensors="pt", padding=True, **kwargs):
        L = max(len(s) for s in seqs) + 2  # +2 for BOS/EOS
        B = len(seqs)
        return {
            "input_ids":      torch.zeros(B, L, dtype=torch.long),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }

    tok.side_effect = _call
    return tok


def _protein_encoder(use_structure: bool = False) -> ProteinEncoder:
    """Build a ProteinEncoder with mocked ESM-2 (no network download)."""
    mock_model = _mock_esm2(SEQ_HIDDEN)
    mock_tok = _mock_tokenizer()
    with patch.object(ProteinEncoder, "_hf_load", return_value=(mock_model, mock_tok)):
        enc = ProteinEncoder(
            latent_dim=LATENT_DIM,
            use_structure=use_structure,
            gvp_hidden_dim=32,
            gvp_layers=1,
            device=torch.device("cpu"),
        )
    return enc


def _sample_coords(L: int = SEQ_LEN) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((L, 3)).astype(np.float32) * 30.0


def _sample_plddt(L: int = SEQ_LEN) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(40.0, 95.0, size=L).astype(np.float32)


# ---------------------------------------------------------------------------
# build_protein_graph
# ---------------------------------------------------------------------------

class TestBuildProteinGraph:
    def test_node_s_shape(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        assert g["node_s"].shape == (SEQ_LEN, 1)

    def test_node_v_shape(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        assert g["node_v"].shape == (SEQ_LEN, 1, 3)

    def test_edge_s_shape(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        E = g["edge_index"].shape[1]
        assert g["edge_s"].shape == (E, 1)

    def test_edge_v_shape(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        E = g["edge_index"].shape[1]
        assert g["edge_v"].shape == (E, 1, 3)

    def test_edge_v_unit_vectors(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        norms = g["edge_v"].norm(dim=-1)  # (E, 1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_plddt_node_features_in_range(self):
        coords = _sample_coords()
        plddt = _sample_plddt()
        g = build_protein_graph(coords, plddt=plddt)
        node_s = g["node_s"]
        assert node_s.min().item() >= 0.0
        assert node_s.max().item() <= 1.0

    def test_no_plddt_gives_ones(self):
        coords = _sample_coords()
        g = build_protein_graph(coords, plddt=None)
        assert torch.all(g["node_s"] == 1.0)

    def test_backward_compat_x(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        assert "x" in g
        assert g["x"].shape == (SEQ_LEN, 3)

    def test_backward_compat_edge_attr(self):
        coords = _sample_coords()
        g = build_protein_graph(coords)
        assert "edge_attr" in g


# ---------------------------------------------------------------------------
# ActionEncoder
# ---------------------------------------------------------------------------

class TestActionEncoder:
    @pytest.fixture
    def ae(self):
        return ActionEncoder(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM,
                             device=torch.device("cpu"))

    def _action(self):
        return {
            "position": torch.tensor(3, dtype=torch.long),
            "aa_old":   torch.tensor(0, dtype=torch.long),
            "aa_new":   torch.tensor(7, dtype=torch.long),
        }

    def test_output_shape_scalar(self, ae):
        z = ae(self._action())
        assert z.shape == (LATENT_DIM,)

    def test_output_shape_batched(self, ae):
        action = {k: v.unsqueeze(0).expand(4) for k, v in self._action().items()}
        z = ae(action)
        assert z.shape == (4, LATENT_DIM)

    def test_position_long_accepted(self, ae):
        action = self._action()
        assert action["position"].dtype == torch.long
        z = ae(action)
        assert not torch.isnan(z).any()

    def test_output_is_finite(self, ae):
        z = ae(self._action())
        assert torch.isfinite(z).all()

    def test_default_mlp_created(self):
        ae = ActionEncoder(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM)
        assert ae.action_mlp is not None

    def test_gradient_flows(self, ae):
        action = {k: v.unsqueeze(0) for k, v in self._action().items()}
        for p in ae.parameters():
            p.requires_grad_(True)
        z = ae(action)
        z.sum().backward()
        grads = [p.grad for p in ae.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# ProteinEncoder (seq-only mode, using mock ESM-2)
# ---------------------------------------------------------------------------

class TestProteinEncoderSeqOnly:
    @pytest.fixture
    def enc(self):
        return _protein_encoder(use_structure=False)

    def test_embed_output_keys(self, enc):
        obs = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        emb = enc.embed_observation(obs)
        assert "seq_emb" in emb
        assert "struct_emb" in emb
        assert "ptm" in emb

    def test_struct_emb_is_none(self, enc):
        obs = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        emb = enc.embed_observation(obs)
        assert emb["struct_emb"] is None

    def test_encode_output_shape(self, enc):
        obs = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        z = enc(obs)
        assert z.shape == (1, LATENT_DIM)

    def test_encode_output_finite(self, enc):
        obs = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        z = enc(obs)
        assert torch.isfinite(z).all()

    def test_forward_same_as_embed_then_encode(self, enc):
        obs = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        z_direct = enc(obs)
        embedded = enc.embed_observation(obs)
        z_indirect = enc.encode(embedded)
        assert torch.allclose(z_direct, z_indirect, atol=1e-6)


# ---------------------------------------------------------------------------
# ProteinEncoder (structure mode, using mock ESM-2 + real GVP-GNN)
# ---------------------------------------------------------------------------

class TestProteinEncoderWithStructure:
    @pytest.fixture
    def enc(self):
        return _protein_encoder(use_structure=True)

    def _obs(self, L: int = SEQ_LEN):
        return {
            "sequence": "A" * L,
            "coords":   _sample_coords(L),
            "plddt":    _sample_plddt(L),
            "ptm":      0.87,
        }

    def test_struct_emb_shape(self, enc):
        obs = self._obs()
        emb = enc.embed_observation(obs)
        assert emb["struct_emb"] is not None
        # (L, gvp_hidden_dim)
        assert emb["struct_emb"].shape[-1] == 32

    def test_ptm_propagated(self, enc):
        obs = self._obs()
        emb = enc.embed_observation(obs)
        assert emb["ptm"] == pytest.approx(0.87)

    def test_output_shape(self, enc):
        z = enc(self._obs())
        assert z.shape == (1, LATENT_DIM)

    def test_output_finite(self, enc):
        z = enc(self._obs())
        assert torch.isfinite(z).all()

    def test_missing_coords_falls_back_gracefully(self, enc):
        obs = {"sequence": "A" * SEQ_LEN}  # no coords
        # Should not raise; struct_emb will be zeros
        z = enc(obs)
        assert z.shape == (1, LATENT_DIM)
        assert torch.isfinite(z).all()

    def test_no_plddt_uses_constant_ones(self, enc):
        obs = self._obs()
        obs.pop("plddt")
        z = enc(obs)
        assert torch.isfinite(z).all()
