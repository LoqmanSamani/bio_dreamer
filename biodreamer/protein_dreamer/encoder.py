"""
biodreamer.protein_dreamer.encoder — Protein Sequence + Structure Encoder.

Purpose:
    Encodes a protein (sequence + predicted/known structure) into a joint
    latent state z_t that captures both evolutionary and geometric information.

Components to implement:
    - ProteinEncoder(BaseEncoder):
        Two-branch encoder fused into a single latent vector:

        Branch 1 — Sequence encoder (ESM-2):
            - Loads pre-trained ESM-2 (650M or 3B) from Hugging Face Hub
            - Input: amino acid sequence string → tokenised → ESM-2 → per-residue embeddings
            - Pooling: mean-pool or [CLS] token → sequence-level embedding

        Branch 2 — Structure encoder (GVP-GNN):
            - Input: backbone coordinates (N, Cα, C, O per residue) from ESMFold/AF2
            - GVP-GNN produces per-residue geometric embeddings
            - Pooling → structure-level embedding

        Fusion: concatenate or cross-attend sequence + structure embeddings → MLP → z_t

    Pre-trained models (from Hugging Face):
        - facebook/esm2_t33_650M_UR50D  (ESM-2 650M)
        - facebook/esm2_t36_3B_UR50D    (ESM-2 3B, for scaling)
        - Custom GVP-GNN checkpoints (trained and uploaded by us)

Design notes:
    - ESM-2 weights can be frozen (feature extraction) or fine-tuned (LoRA adapters).
    - Structure encoder is optional — ProteinDreamer v1 can work sequence-only,
      then structure is added in v2 (natural PhD progression).
    - The encoder runs once per protein state, so moderate latency is acceptable.
"""
import torch
import torch.nn as nn
from biodreamer.core.encoder import BaseEncoder
from typing import Any, Optional, Dict





class ProteinEncoder(BaseEncoder):
    """Encodes a protein (sequence + structure(optional)) into a latent state z_t."""
    def __init__(
        self, 
        latent_dim: int, 
        fusion_mlp: Any, 
        sequence_encoder: Optional[Any] = None, 
        structure_encoder: Optional[Any] = None ,
        regularizer: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__(latent_dim)
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.latent_dim = latent_dim
        self.sequence_encoder = sequence_encoder.to(self.device)
        self.fusion_mlp = fusion_mlp.to(self.device)
        self.structure_encoder = structure_encoder.to(self.device) if structure_encoder is not None else None
        self.regularizer = regularizer.to(self.device) if regularizer is not None else None
        self.layer_norm = nn.LayerNorm(self.latent_dim).to(self.device)
        
    def encode(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Encode a protein observation into a latent space z_t."""
        seq_emb = observation['seq_emb'].to(self.device)
        struct_emb = observation.get('struct_emb', None)
        
        if len(seq_emb.shape) == 3:
            seq_emb = seq_emb.mean(dim=1)
        elif len(seq_emb.shape) == 2:
            seq_emb = seq_emb.mean(dim=0)
        else:
            raise ValueError(f"Unexpected shape for seq_emb: {seq_emb.shape}")
        if struct_emb is not None:
            struct_emb = struct_emb.to(self.device)
            if len(struct_emb.shape) == 3:
                struct_emb = struct_emb.mean(dim=1)
            elif len(struct_emb.shape) == 2:
                struct_emb = struct_emb.mean(dim=0)
            else:
                raise ValueError(f"Unexpected shape for struct_emb: {struct_emb.shape}")
            
        if struct_emb is not None:
            info_embed = torch.cat([seq_emb, struct_emb], dim=-1)
        else:
            info_embed = seq_emb
            
        z_t = self.layer_norm(self.fusion_mlp(info_embed))
        
        if self.regularizer is not None:
            z_t = self.regularizer(z_t)
            
        return z_t
            
        
    def embed_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Embed the raw observation into sequence and structure embeddings."""
        seq_emb = self.sequence_encoder(observation['sequence'])
        struct_emb = self.structure_encoder(observation['structure']) if self.structure_encoder is not None else None
        return {'seq_emb': seq_emb, 'struct_emb': struct_emb}
    
    
    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Full forward pass: embed observation(optional, if input is raw) and then encode to z_t."""
        if 'sequence' in observation:
            embedded_obs = self.embed_observation(observation)
        else:
            embedded_obs = observation
        return self.encode(embedded_obs)
    
    
    
    
class ActionEncoder(BaseEncoder):
    """Encodes an action (e.g. mutation) into a latent space z_t."""
    def __init__(
        self, 
        latent_dim: int, 
        action_mlp: Any, 
        pos_embed: Optional[Any] = None, 
        aa_embed: Optional[Any] = None, 
        aa_new_embed: Optional[Any] = None, 
        device: Optional[torch.device] = None,
        embed_dim: int = 128
        ) -> None:
        super().__init__(latent_dim)
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.latent_dim = latent_dim
        self.action_mlp = action_mlp.to(self.device)
        self.pos_embed = pos_embed.to(self.device) if pos_embed is not None else nn.Linear(1, embed_dim).to(self.device)
        self.aa_embed = aa_embed.to(self.device) if aa_embed is not None else nn.Embedding(20, embed_dim).to(self.device)
        self.aa_new_embed = aa_new_embed.to(self.device) if aa_new_embed is not None else nn.Embedding(20, embed_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(self.latent_dim).to(self.device)
        
    def encode(self, action: Dict[str, Any]) -> torch.Tensor:
        """Encode an action into a latent space z_t."""
        
        pos_emb = self.pos_embed(action['position'].unsqueeze(-1).to(self.device))
        aa_emb = self.aa_embed(action['aa_old'].to(self.device))
        aa_new_emb = self.aa_new_embed(action['aa_new'].to(self.device))
        action_embed = torch.cat([pos_emb, aa_emb, aa_new_emb], dim=-1)
        
        return self.layer_norm(self.action_mlp(action_embed))
            
    def forward(self, action: Dict[str, Any]) -> torch.Tensor:
        """Full forward pass: encode action to z_t."""
        return self.encode(action)  