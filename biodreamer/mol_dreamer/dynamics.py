"""
biodreamer.mol_dreamer.dynamics — Latent Dynamics Model for Molecular Systems.

Purpose:
    Predicts how the latent state of a molecular system evolves over time:
        z_{t+1} = f_θ(z_t, a_t, Δt)
    where a_t may be an external perturbation (mutation, force, temperature change)
    and Δt allows variable time steps (coarse-grained jumps).

Components to implement:
    - MolDynamics(BaseDynamics):
        Options for the dynamics backbone:
        1. Latent Diffusion: denoising diffusion model that generates z_{t+1}
           conditioned on z_t and a_t. Captures multi-modal transitions.
        2. Neural SDE: stochastic differential equation in latent space,
           solved with torchdiffeq. Naturally models Brownian dynamics.
        3. RSSM (DreamerV3-style): deterministic path + stochastic state,
           trained with KL-balanced variational objective.

Design notes:
    - Variable time steps are critical for MD: the dynamics model should support
      both fine-grained (femtosecond) and coarse-grained (nanosecond) jumps.
    - Stochastic transitions are physically motivated (thermal fluctuations).
    - The dynamics model's uncertainty over z_{t+1} feeds into active inference
      for exploration-driven molecular design.
"""
