"""
biodreamer.core.active_inference — Active Inference / Free Energy Principle module.

Purpose:
    Implements the Active Inference framework for action selection, which is
    the theoretical backbone of ProteinDreamer (and optionally the other modules).

    Active Inference replaces the standard RL objective (maximise expected reward)
    with minimisation of Expected Free Energy (EFE):

        G(π) = E_π[ D_KL[q(o|π) || p(o)] - H[q(o|s,π)] ]
             = E_π[ Risk ] - E_π[ Information Gain ]
             = pragmatic_value + epistemic_value

    This naturally balances:
        - Exploitation (pragmatic value): seek states with high reward / low surprise
        - Exploration (epistemic value): seek states that reduce model uncertainty

Components to implement:
    - ExpectedFreeEnergy:
        - compute_efe(world_model, z_t, action_candidates) → EFE scores per action
        - pragmatic_value(z_t, action) → expected reward under policy
        - epistemic_value(z_t, action) → expected information gain (uncertainty reduction)

    - ActiveInferencePolicy(BasePolicy):
        - Selects actions by minimising EFE over candidate actions
        - Uses the world model's predictive uncertainty for epistemic value
        - Supports both discrete (mutation) and continuous (latent) action spaces

    - BeliefUpdater:
        - Updates the agent's beliefs (posterior over world model parameters or
          latent states) after receiving new observations
        - Connects to the active learning loop (new experimental data → belief update)

Design notes:
    - The EFE decomposition into pragmatic + epistemic value is the key novelty
      of ProteinDreamer vs. standard RL approaches (DynaPPO, EvoPlay, μSearch).
    - Epistemic value requires uncertainty estimates from the dynamics model
      (ensemble disagreement, evidential uncertainty, or Bayesian posterior).
    - Reference: Parr, Pezzulo & Friston (2022) "Active Inference" MIT Press;
      Tschantz et al. (2020) "RL Through Active Inference."
"""
