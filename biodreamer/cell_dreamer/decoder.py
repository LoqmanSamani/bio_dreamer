"""
biodreamer.cell_dreamer.decoder — Gene Expression Profile Decoder.

Purpose:
    Reconstructs full gene expression profiles from latent cell states.
    Provides the interpretable output that biologists interact with.

Components to implement:
    - CellDecoder(BaseDecoder):
        - decode(z_t) → gene expression vector (N_genes,)
        - decode_with_uncertainty(z_t) → (mean, variance) per gene
        - decode_top_genes(z_t, k) → top-k differentially expressed genes

    Architecture:
        - MLP decoder mapping z_t → gene expression (matching scVI-style decoder)
        - Zero-inflated negative binomial (ZINB) output distribution for count data
        - Library size correction (decoder accounts for sequencing depth)

Design notes:
    - The decoder reconstructs ALL genes, even though the encoder may use only
      HVGs. Non-HVG reconstruction uses the latent state + a wider decoder.
    - Output is used for the web frontend's gene expression heatmap and
      differential expression analysis.
    - Loss: negative log-likelihood under ZINB distribution (standard for scRNA-seq).
"""
