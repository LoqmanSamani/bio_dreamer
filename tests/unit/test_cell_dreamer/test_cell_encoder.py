"""
test_cell_encoder.py — Unit Tests for biodreamer.cell_dreamer.encoder.

Tests to implement:
    - test_scvi_encoder_output: VAE produces latent z and library size
    - test_batch_correction: different batch labels produce corrected embeddings
    - test_zinb_parameterisation: output parameters are valid for ZINB distribution
    - test_variable_gene_counts: handles different numbers of highly-variable genes
    - test_sparse_input: works with scipy sparse matrices
"""
