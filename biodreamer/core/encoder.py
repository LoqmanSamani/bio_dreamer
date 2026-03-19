"""
biodreamer.core.encoder — Base encoder interface.

Purpose:
    Abstract base class for all domain-specific encoders. An encoder maps
    raw observations (atomic coordinates, protein sequences+structures,
    gene expression profiles) to a compact latent representation z_t.

Components to implement:
    - BaseEncoder(nn.Module):
        - encode(observation) → z_t (latent state tensor)
        - get_latent_dim() → int

    Domain-specific encoders inherit from this:
        - MolDreamer:     SE(3)-equivariant GNN (EGNN, PaiNN) on atomic graphs
        - ProteinDreamer: ESM-2 (sequence) + GVP-GNN (structure) → joint embedding
        - CellDreamer:    VAE encoder on scRNA-seq count vectors

Design notes:
    - Encoders may wrap pre-trained models from Hugging Face (e.g., ESM-2).
      The base class should support freezing/unfreezing pre-trained weights.
    - Output latent states should be standardised (zero mean, unit variance)
      for stable dynamics model training.
"""
