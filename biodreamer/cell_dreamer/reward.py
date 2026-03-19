"""
biodreamer.cell_dreamer.reward — Cell State Objective Function.

Purpose:
    Computes reward based on how close the predicted cell state is to a
    user-defined target phenotype (e.g., target cell type, apoptotic state,
    high metabolite production).

Components to implement:
    - CellRewardHead(BaseRewardHead):
        - set_target(target_expression_profile) → precompute target embedding
        - predict(z_t) → scalar reward (distance to target in latent space)
        - predict_multi(z_t) → dict with multiple cell state metrics

    Distance metrics:
        - Cosine similarity in latent space
        - Earth mover's distance (Wasserstein) between expression distributions
        - Pearson/Spearman correlation of gene expression profiles
        - Marker gene activation score (user specifies key genes that must be up/down)

Design notes:
    - The target cell state can be specified as:
      1. A reference expression profile (e.g., "I want cells that look like cardiomyocytes")
      2. A set of marker gene constraints (e.g., "TNNT2 high, MYH7 high, OCT4 low")
      3. A latent space region (cluster ID from the training data)
    - The web frontend lets users upload a target profile or select from known
      cell types in a UMAP visualisation.
"""
