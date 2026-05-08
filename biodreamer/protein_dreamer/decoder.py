from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from biodreamer.core.decoder import BaseDecoder
from .blocks import TransformerLayer
from .utils import aa_indices_to_sequence

logger = logging.getLogger(__name__)





class ProteinSequenceDecoder(BaseDecoder):
    """decodes a pooled latent z_t to per-residue amino acid logits.

    architecture: learned positional query embeddings cross-attend to z_t via a
    stack of TransformerLayer blocks, then project to 20-class logits.

    decode() returns:
      "logits":    (B, L, 20) — raw logits, use for cross-entropy training
      "sequences": List[str]  — greedy-decoded aa strings (length L each)

    set_seq_len() updates default_seq_len before each trajectory so that
    WorldModel.decode(z_t) works without an explicit seq_len argument.
    """
    def __init__(
        self,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> None:
        latent_dim      = config["latent_dim"]
        n_layers        = config.get("n_layers", 4)
        n_heads         = config.get("n_heads", 8)
        mlp_ratio       = config.get("mlp_ratio", 4.0)
        dropout         = config.get("dropout", 0.0)
        max_seq_len     = config.get("max_seq_len", 512)
        default_seq_len = config.get("default_seq_len", 50)
        super().__init__(latent_dim)
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.default_seq_len = default_seq_len

        self.pos_queries = nn.Parameter(torch.zeros(max_seq_len, latent_dim))
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

        # z_t (global) is projected to a single key/value token for cross-attention
        self.z_proj = nn.Linear(latent_dim, latent_dim)

        self.layers = nn.ModuleList([
            TransformerLayer({"dim": latent_dim, "n_heads": n_heads, "mlp_ratio": mlp_ratio, "dropout": dropout})
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.logit_head = nn.Linear(latent_dim, 20)

        self.to(self.device)

    def set_seq_len(self, seq_len: int) -> None:
        """update the default sequence length used when seq_len is not passed to decode()"""
        self.default_seq_len = seq_len

    def decode(
        self,
        z_t: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """decode z_t to per-residue sequence logits.

        z_t:     (B, latent_dim)
        seq_len: number of residues to generate; defaults to self.default_seq_len

        returns:
            "logits":    (B, L, 20)
            "sequences": List[str] of length B, each string of length L
        """
        L = seq_len if seq_len is not None else self.default_seq_len
        if L > self.max_seq_len:
            raise ValueError(f"seq_len={L} exceeds max_seq_len={self.max_seq_len}")

        z_t = z_t.to(self.device)
        B = z_t.shape[0]

        # positional queries: (B, L, latent_dim)
        queries = self.pos_queries[:L].unsqueeze(0).expand(B, -1, -1)

        # z_t as single conditioning token: (B, 1, latent_dim)
        cond = self.z_proj(z_t).unsqueeze(1)

        x = queries
        for layer in self.layers:
            x = layer(x, cond=cond)
        x = self.norm(x)                          # (B, L, latent_dim)
        logits = self.logit_head(x)               # (B, L, 20)

        tokens = logits.argmax(dim=-1)            # (B, L)
        sequences: List[str] = [
            aa_indices_to_sequence(tokens[i].cpu()) for i in range(B)
        ]

        return {"logits": logits, "sequences": sequences}

    def forward(self, z_t: torch.Tensor, seq_len: Optional[int] = None) -> Dict[str, Any]:
        return self.decode(z_t, seq_len=seq_len)


class ProteinStructureDecoder(BaseDecoder):
    """decodes a pooled latent z_t to per-residue Cα coordinates.

    architecture: learned positional query embeddings cross-attend to z_t via a
    stack of TransformerLayer blocks, then project to 3D Cα positions.

    intended for interpretability and validation, not a substitute for a
    structure-prediction oracle (ESMFold / AlphaFold).

    decode() returns:
      "coords": (B, L, 3) — predicted Cα coordinates in Å

    set_seq_len() mirrors ProteinSequenceDecoder for consistent usage.
    """
    def __init__(
        self,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> None:
        latent_dim      = config["latent_dim"]
        n_layers        = config.get("n_layers", 4)
        n_heads         = config.get("n_heads", 8)
        mlp_ratio       = config.get("mlp_ratio", 4.0)
        dropout         = config.get("dropout", 0.0)
        max_seq_len     = config.get("max_seq_len", 512)
        default_seq_len = config.get("default_seq_len", 50)
        super().__init__(latent_dim)
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.default_seq_len = default_seq_len

        self.pos_queries = nn.Parameter(torch.zeros(max_seq_len, latent_dim))
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

        self.z_proj = nn.Linear(latent_dim, latent_dim)

        self.layers = nn.ModuleList([
            TransformerLayer({"dim": latent_dim, "n_heads": n_heads, "mlp_ratio": mlp_ratio, "dropout": dropout})
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.coord_head = nn.Linear(latent_dim, 3)

        self.to(self.device)

    def set_seq_len(self, seq_len: int) -> None:
        """update the default sequence length used when seq_len is not passed to decode()"""
        self.default_seq_len = seq_len

    def decode(
        self,
        z_t: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """decode z_t to per-residue Cα coordinates.

        z_t:     (B, latent_dim)
        seq_len: number of residues; defaults to self.default_seq_len

        returns:
            "coords": (B, L, 3) predicted Cα positions in Å
        """
        L = seq_len if seq_len is not None else self.default_seq_len
        if L > self.max_seq_len:
            raise ValueError(f"seq_len={L} exceeds max_seq_len={self.max_seq_len}")

        z_t = z_t.to(self.device)
        B = z_t.shape[0]

        queries = self.pos_queries[:L].unsqueeze(0).expand(B, -1, -1)
        cond = self.z_proj(z_t).unsqueeze(1)

        x = queries
        for layer in self.layers:
            x = layer(x, cond=cond)
        x = self.norm(x)
        coords = self.coord_head(x)               # (B, L, 3)

        return {"coords": coords}

    def forward(self, z_t: torch.Tensor, seq_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        return self.decode(z_t, seq_len=seq_len)
