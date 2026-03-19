"""
biodreamer.core.decoder — Base decoder interface.

Purpose:
    Abstract base class for decoders that reconstruct observable quantities
    from latent states. Used for:
        1. Training signal: reconstruction loss grounds the latent space.
        2. Interpretability: users see predicted structures, sequences, expression profiles.
        3. Validation: decoded outputs can be checked against ground-truth oracles.

Components to implement:
    - BaseDecoder(nn.Module):
        - decode(z_t) → reconstructed observation
        - decode_batch(z_batch) → batch of reconstructed observations

    Domain-specific decoders:
        - MolDreamer:     Decodes latent → 3D atomic coordinates + contact maps + RMSD/RMSF
        - ProteinDreamer: Decodes latent → mutated sequence, predicted backbone, pLDDT map
        - CellDreamer:    Decodes latent → gene expression profile (counts / log-normalised)

Design notes:
    - Decoders may produce multiple output heads (e.g., ProteinDreamer decodes
      both sequence probabilities and structure coordinates).
    - For the web frontend, decoder outputs are the primary visualisation data.
"""
