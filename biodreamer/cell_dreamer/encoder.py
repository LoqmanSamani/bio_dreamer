"""
biodreamer.cell_dreamer.encoder — scRNA-seq Cell State Encoder.

Purpose:
    Encodes single-cell gene expression profiles into a compact latent
    representation z_t that captures the cell's transcriptomic state.

Components to implement:
    - CellEncoder(BaseEncoder):
        VAE-based encoder for scRNA-seq count data:
        - Input: gene expression vector (N_genes,) — log-normalised counts
        - Architecture: MLP encoder → (μ, log σ²) → reparameterisation → z_t
        - Output: latent cell state z_t ∈ R^d (d = 64–256)

        Alternative architecture options:
        - scVI-style encoder with library-size correction and zero-inflation
        - Transformer encoder (scGPT-style) for gene-token representations
        - Pre-trained single-cell foundation model (scGPT, Geneformer from HF Hub)

Design notes:
    - Gene selection: encode only highly variable genes (HVGs, 2000–5000) to
      reduce dimensionality. Full genome reconstruction via decoder.
    - The VAE latent space should be continuous and smooth for the Neural ODE
      dynamics model to operate on.
    - Batch effect correction: include batch/condition embeddings to prevent
      the latent space from encoding technical artefacts instead of biology.
"""
