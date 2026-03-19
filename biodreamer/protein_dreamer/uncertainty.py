"""
biodreamer.protein_dreamer.uncertainty — Epistemic Uncertainty Estimation.

Purpose:
    Estimates the world model's uncertainty about its predictions — critical
    for active inference (exploration) and active learning (deciding which
    mutants to test experimentally).

Components to implement:
    - EnsembleUncertainty:
        Maintains an ensemble of N world models (or N reward heads).
        Uncertainty = disagreement (variance) across ensemble members.
        Pros: simple, well-understood. Cons: N× memory and compute.

    - EvidentialUncertainty:
        Single model that outputs parameters of a higher-order distribution
        (e.g., Normal-Inverse-Gamma for regression). Estimates both aleatoric
        and epistemic uncertainty in a single forward pass.
        Pros: efficient. Cons: can be miscalibrated.

    - MCDropoutUncertainty:
        Uses dropout at inference time (Monte Carlo Dropout) to approximate
        Bayesian posterior. N stochastic forward passes → variance.
        Pros: easy to implement. Cons: often underestimates uncertainty.

    - UncertaintyModule (unified interface):
        - estimate(z_t, action) → (mean_prediction, epistemic_uncertainty, aleatoric_uncertainty)
        - calibrate(validation_data) → calibrate uncertainty estimates
        - is_uncertain(z_t, action, threshold) → bool

Design notes:
    - The active inference module (core/active_inference.py) uses epistemic
      uncertainty to compute the information gain (epistemic value) component
      of Expected Free Energy.
    - The active learning loop (training/active_learning.py) uses uncertainty
      to select the most informative candidates for experimental validation.
    - Uncertainty should be calibrated — predicted confidence intervals should
      match empirical coverage. Evaluate with calibration plots.
"""
