"""
biodreamer.cell_dreamer.data.preprocessing — scRNA-seq Preprocessing Utilities.

Purpose:
    Standard single-cell RNA-seq preprocessing pipeline and perturbation
    data formatting for CellDreamer training.

Components to implement:
    - load_anndata(h5ad_path) → AnnData object
    - qc_filter(adata, min_genes, min_cells, max_mito) → filtered AnnData
    - normalise(adata, target_sum) → log-normalised AnnData
    - select_hvg(adata, n_top_genes) → AnnData with highly variable genes
    - compute_pca(adata, n_components) → PCA-reduced AnnData
    - compute_umap(adata) → AnnData with UMAP coordinates
    - extract_perturbation_pairs(adata) → list of (control, perturbation, outcome) tuples
    - encode_perturbation(perturbation_id, gene_list) → perturbation tensor
    - batch_correct(adata, method) → batch-corrected AnnData (Harmony, scVI, etc.)
    - split_train_val_test(adata, strategy) → (train, val, test) AnnData splits

Design notes:
    - Follows scanpy's standard preprocessing workflow.
    - Perturbation extraction groups cells by guide RNA identity and computes
      per-perturbation mean expression profiles.
    - Support for both cell-level (individual cells) and pseudobulk
      (aggregated per perturbation) training modes.
"""
