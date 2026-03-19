"""
biodreamer.core.reward — Base reward head interface.

Purpose:
    Abstract base class for reward prediction from latent states. The reward
    head maps a latent state z_t to a scalar (or multi-dimensional) reward
    signal that the RL policy optimises.

Components to implement:
    - BaseRewardHead(nn.Module):
        - predict(z_t) → reward (scalar tensor)
        - predict_multi(z_t) → dict of named rewards (for multi-objective)

    Domain-specific reward heads:
        - MolDreamer:     Predicts binding ΔG, thermostability Tm, SASA
        - ProteinDreamer: Predicts ΔΔG stability, Kd affinity, kcat activity,
                          expressibility — trained on DMS data
        - CellDreamer:    Predicts distance to target cell state (cosine similarity,
                          Wasserstein distance to target expression profile)

Design notes:
    - Multi-objective rewards should support configurable scalarisation
      (weighted sum, Tchebycheff, hypervolume-based).
    - Reward heads are trained jointly with the world model on experimental
      fitness data. They must output calibrated uncertainties for active inference.
"""
