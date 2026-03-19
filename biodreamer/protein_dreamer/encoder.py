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
