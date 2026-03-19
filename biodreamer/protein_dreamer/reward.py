"""
biodreamer.protein_dreamer.reward — Multi-Objective Protein Fitness Predictor.

Purpose:
    Predicts fitness scores from protein latent states — the reward signal
    that the RL policy optimises. Supports multi-objective optimisation
    (stability + function + expressibility simultaneously).

Components to implement:
    - ProteinRewardHead(BaseRewardHead):
        Multi-task prediction heads branching from the shared latent state:
        - predict_stability(z_t) → ΔΔG (kcal/mol) — thermodynamic stability change
        - predict_affinity(z_t) → Kd or IC50 — binding affinity
        - predict_activity(z_t) → kcat or relative activity — catalytic function
        - predict_expressibility(z_t) → solubility / expression level score
        - predict_multi(z_t) → dict of all fitness dimensions
        - scalarise(rewards_dict, weights) → single scalar reward

    Training data:
        - ProteinGym DMS assays (200+ proteins, diverse fitness measurements)
        - Tsuboyama mega-scale ΔΔG data (500k+ measurements)
        - BRENDA / EnzML enzyme activity data

    Multi-objective scalarisation strategies:
        - Weighted linear sum (configurable weights in YAML)
        - Tchebycheff scalarisation
        - Pareto-based (return full reward vector, let policy handle trade-offs)

Design notes:
    - Each head should output both a point prediction and an uncertainty estimate.
    - The reward head is fine-tuned per protein family for maximum accuracy.
    - Reward shaping: optionally add a novelty bonus (distance from wild-type in
      latent space) to encourage diverse exploration.
"""
