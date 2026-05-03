from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.encoder import BaseEncoder

logger = logging.getLogger(__name__)





class ProteinEncoder(BaseEncoder):
    """encodes a protein (sequence + optional structure) into a latent state z_t.

    sequence branch: esm2 (frozen by default). sequence embedding is mean-pooled
    over residue positions (BOS/EOS tokens are stripped before pooling).

    structure branch (use_structure=True): GVP-GNN operating on a Cα contact graph.
    Node features: pLDDT (scalar) + Cα position (vector). edge features: distance
    (scalar) + unit displacement (vector). structural embedding is mean-pooled over L.

    fusion: [seq_pooled | struct_pooled | pTM] -> MLP -> LayerNorm -> z_t.
    when use_structure=False, only seq_pooled is used.
    """

    # esm2 hidden sizes by shorthand name
    _ESM2_HIDDEN: Dict[str, int] = {
        "esm2-8m":   320,
        "esm2-35m":  480,
        "esm2-150m": 640,
        "esm2-650m": 1280,
        "esm2-3b":   2560,
        "esm2-15b":  5120,
    }

    def __init__(
        self,
        latent_dim: int,
        sequence_encoder: Optional[Any] = None,
        sequence_tokenizer: Optional[Any] = None,
        structure_encoder: Optional[Any] = None,
        freeze_seq_encoder: bool = True,
        freeze_struct_encoder: bool = True,
        use_structure: bool = False,
        seq_model_name: str = "esm2-650m",
        gvp_hidden_dim: int = 256,
        gvp_layers: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(latent_dim)
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.freeze_seq_encoder = freeze_seq_encoder
        self.freeze_struct_encoder = freeze_struct_encoder
        self.use_structure = use_structure
        self.gvp_hidden_dim = gvp_hidden_dim

        # sequence encoder (ESM-2)
        if sequence_encoder is not None:
            self.sequence_encoder = sequence_encoder.to(self.device)
            self.sequence_tokenizer = sequence_tokenizer
        else:
            esm2, seq_tok = self._hf_load(seq_model_name)
            self.sequence_encoder = esm2.to(self.device)
            self.sequence_tokenizer = seq_tok

        seq_hidden: int = getattr(
            self.sequence_encoder.config, "hidden_size",
            self._ESM2_HIDDEN.get(seq_model_name, 1280),
        )

        if freeze_seq_encoder:
            for p in self.sequence_encoder.parameters():
                p.requires_grad_(False)

        # structure encoder (GVP-GNN)
        if use_structure:
            if structure_encoder is not None:
                self.structure_encoder = structure_encoder.to(self.device)
            else:
                from .nets import GVP_GNN
                self.structure_encoder = GVP_GNN(
                    in_node_dims=(1, 1),            # (plddt scalar, Cα position vector)
                    in_edge_dims=(1, 1),            # (distance scalar, unit-displacement vector)
                    hidden_dims=(gvp_hidden_dim, 4),
                    n_layers=gvp_layers,
                ).to(self.device)
            if freeze_struct_encoder:
                for p in self.structure_encoder.parameters():
                    p.requires_grad_(False)
        else:
            self.structure_encoder = None

        # fusion MLP
        fusion_in = seq_hidden + (gvp_hidden_dim + 1 if use_structure else 0)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        ).to(self.device)
        self.layer_norm = nn.LayerNorm(latent_dim).to(self.device)

    def embed_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """embed a raw observation dict -> {seq_emb, struct_emb, ptm}.

        expected observation keys:
        - 'sequence' (str): amino acid sequence
        - 'coords' (np.ndarray, optional): (L, 3) Cα coordinates (required if use_structure)
        - 'plddt' (np.ndarray, optional): per-residue confidence (L,)
        - 'ptm' (float, optional): global pTM score
        """
        # sequence embedding
        inputs = self.sequence_tokenizer(
            [observation["sequence"]], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.freeze_seq_encoder:
            with torch.no_grad():
                seq_emb = self.sequence_encoder(**inputs).last_hidden_state  # (1, L+2, H)
        else:
            seq_emb = self.sequence_encoder(**inputs).last_hidden_state

        out: Dict[str, Any] = {"seq_emb": seq_emb, "struct_emb": None, "ptm": None}
        # structure embedding
        if self.use_structure and self.structure_encoder is not None:
            coords = observation.get("coords")
            plddt = observation.get("plddt")
            out["ptm"] = observation.get("ptm")

            if coords is not None:
                from .data.preprocessing import build_protein_graph
                graph = build_protein_graph(coords, plddt=plddt)
                node_s = graph["node_s"].to(self.device)      # (L, 1)
                node_v = graph["node_v"].to(self.device)      # (L, 1, 3)
                edge_index = graph["edge_index"].to(self.device)
                edge_s = graph["edge_s"].to(self.device)      # (E, 1)
                edge_v = graph["edge_v"].to(self.device)      # (E, 1, 3)

                if self.freeze_struct_encoder:
                    with torch.no_grad():
                        struct_s, _ = self.structure_encoder(
                            node_s, node_v, edge_index, edge_s, edge_v
                        )
                else:
                    struct_s, _ = self.structure_encoder(
                        node_s, node_v, edge_index, edge_s, edge_v
                    )
                out["struct_emb"] = struct_s  # (L, gvp_hidden_dim)
            else:
                logger.warning(
                    "use_structure=True but 'coords' not in observation; structure encoder skipped"
                )

        return out

    def encode(self, observation: Dict[str, Any]) -> torch.Tensor:
        """fuse pre-computed embeddings → z_t of shape (1, latent_dim)."""
        seq_emb = observation["seq_emb"].to(self.device)

        # pool sequence: (1, L+2, H) → strip BOS/EOS -> mean over L -> (1, H)
        if seq_emb.dim() == 3:
            seq_emb = seq_emb[:, 1:-1, :].mean(dim=1)
        elif seq_emb.dim() == 2:
            seq_emb = seq_emb.mean(dim=0, keepdim=True)

        parts = [seq_emb]
        if self.use_structure:
            struct_emb = observation.get("struct_emb")
            ptm = observation.get("ptm")

            if struct_emb is not None:
                struct_emb = struct_emb.to(self.device)
                if struct_emb.dim() == 2:
                    struct_emb = struct_emb.mean(dim=0, keepdim=True)  # (L, H) -> (1, H)
                parts.append(struct_emb)
            else:
                # graceful degradation: zeros when structure was unavailable
                parts.append(torch.zeros(seq_emb.shape[0], self.gvp_hidden_dim, device=self.device))

            ptm_t = torch.tensor(
                [[ptm if ptm is not None else 0.0]], dtype=torch.float32, device=self.device
            )
            parts.append(ptm_t)

        fused = torch.cat(parts, dim=-1)
        return self.layer_norm(self.fusion_mlp(fused))

    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Full forward pass: embed raw observation (if needed) then encode to z_t."""
        if "sequence" in observation:
            observation = self.embed_observation(observation)
        return self.encode(observation)

    def _hf_load(self, model_name: str) -> tuple:
        from .model_loader import HFModelLoader
        loader = HFModelLoader(device=self.device)
        return loader.load(model_name)


class ActionEncoder(BaseEncoder):
    """encodes a single-site mutation (position, wt aa, mut aa) into a latent action vector.

    input dict keys:
    - 'position' (torch.long): 0-based residue index
    - 'aa_old'   (torch.long): wild-type amino acid index (0–19)
    - 'aa_new'   (torch.long): mutant amino acid index (0–19)
    """
    def __init__(
        self,
        latent_dim: int,
        action_mlp: Optional[Any] = None,
        pos_embed: Optional[Any] = None,
        aa_embed: Optional[Any] = None,
        aa_new_embed: Optional[Any] = None,
        device: Optional[torch.device] = None,
        embed_dim: int = 128,
    ) -> None:
        super().__init__(latent_dim)
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.latent_dim = latent_dim
        self.pos_embed = pos_embed.to(self.device) if pos_embed is not None else nn.Linear(
            1, embed_dim).to(self.device)
        self.aa_embed = aa_embed.to(self.device) if aa_embed is not None else nn.Embedding(
            20, embed_dim).to(self.device)
        self.aa_new_embed = aa_new_embed.to(self.device) if aa_new_embed is not None else nn.Embedding(
            20, embed_dim).to(self.device)
        self.action_mlp = action_mlp.to(self.device) if action_mlp is not None else nn.Sequential(
            nn.Linear(3 * embed_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        ).to(self.device)
        self.layer_norm = nn.LayerNorm(latent_dim).to(self.device)

    def encode(self, action: Dict[str, Any]) -> torch.Tensor:
        pos = action["position"].to(self.device)
        pos_emb = self.pos_embed(pos.float().unsqueeze(-1))
        aa_emb = self.aa_embed(action["aa_old"].float().to(self.device))
        aa_new_emb = self.aa_new_embed(action["aa_new"].float().to(self.device))
        action_embed = torch.cat([pos_emb, aa_emb, aa_new_emb], dim=-1)
        return self.layer_norm(self.action_mlp(action_embed))

    def forward(self, action: Dict[str, Any]) -> torch.Tensor:
        return self.encode(action)
