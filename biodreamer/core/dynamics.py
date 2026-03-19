"""
biodreamer.core.dynamics — Base dynamics model interface.

Purpose:
    Abstract base class for latent dynamics models. A dynamics model predicts
    how the latent state evolves after an action is taken:
        z_{t+1} = f_theta(z_t, a_t)

    This is the "world model" in the narrow sense — it learns the transition
    function of the biological environment in latent space.

Components to implement:
    - BaseDynamics(nn.Module):
        - predict(z_t, action) → z_{t+1} (next latent state)
        - predict_distribution(z_t, action) → distribution over z_{t+1}
          (for uncertainty estimation and active inference)
        - rollout(z_0, actions_sequence) → list of z_t states

    Domain-specific dynamics models:
        - MolDreamer:     Latent diffusion / neural SDE for molecular coordinate evolution
        - ProteinDreamer: Conditional transformer or latent diffusion predicting
                          structure+fitness shifts after mutations
        - CellDreamer:    Neural ODE/SDE for temporal gene regulatory dynamics

Design notes:
    - Support both deterministic and stochastic transitions (RSSM-style).
    - The stochastic component is critical for active inference: the agent
      needs to estimate uncertainty over next states to compute epistemic value.
    - Support variable time steps (especially for CellDreamer Neural ODE and
      MolDreamer coarse-grained jumps).
"""
