"""
biodreamer.utils.metrics — Evaluation Metrics.

Purpose:
    Standardised evaluation metrics for assessing world model accuracy,
    fitness prediction quality, and design success across all three modules.

Components to implement:
    Fitness prediction metrics:
        - spearman_rho(predicted, actual) → Spearman rank correlation
        - ndcg_at_k(predicted, actual, k) → NDCG for top-k variant ranking
        - mse(predicted, actual) → mean squared error
        - pearson_r(predicted, actual) → Pearson correlation

    World model evaluation:
        - rollout_consistency(model, data, horizon) → multi-step prediction error
        - calibration_error(predicted_mean, predicted_std, actual) → ECE

    Design campaign metrics:
        - max_fitness_found(candidates, ground_truth) → best fitness discovered
        - diversity_score(candidates) → average pairwise distance in sequence/latent space
        - efficiency(n_evaluations, fitness_achieved) → fitness per evaluation (for active learning)

    ProteinGym benchmark:
        - proteingym_eval(model, assay_id) → Spearman ρ on specific DMS assay
        - proteingym_summary(model) → average Spearman over all ProteinGym assays

Design notes:
    - All metrics return plain Python floats (JSON-serialisable for the API).
    - Metric names follow ProteinGym conventions for easy comparison with leaderboard.
"""
