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
    def __init__(self, latent_dim: int, sequence_model: Any, structure_model: Optional[Any] = None, device: Optional[torch.device] = None) -> None:
        super().__init__(latent_dim)
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.latent_dim = latent_dim
        self.sequence_model = sequence_model.to(self.device)
        self.structure_model = structure_model.to(self.device) if structure_model is not None else None
        
    def encode(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Encode a protein observation into a latent space z_t."""
        sequence = observation['sequence'].to(self.device)
        structure = observation.get('structure', None)
        seq_emb = self.encode_sequence(sequence)
        
        if self.structure_model is not None and structure is not None:
            structure = structure.to(self.device)
            struct_emb = self.encode_structure(structure)
            combined_emb = torch.cat([seq_emb, struct_emb], dim = -1)
            return combined_emb
        else:
            return seq_emb
            

