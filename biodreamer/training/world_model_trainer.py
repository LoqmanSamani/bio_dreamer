"""
biodreamer.training.world_model_trainer — World Model Training Loop.

Purpose:
    Trains the full world model (encoder + dynamics + decoder + reward head)
    on domain-specific data. This is the primary training loop — the world
    model must be trained before the RL policy.

Components to implement:
    - WorldModelTrainer(BaseTrainer):
        Training procedure (following DreamerV3):
        1. Encode observations → latent states z_t
        2. Train dynamics model: predict z_{t+1} from (z_t, action)
        3. Train decoder: reconstruct observation from z_{t+1}
        4. Train reward head: predict fitness/property from z_t
        5. KL loss: regularise latent space (for VAE-based encoders)

        Loss = reconstruction_loss + dynamics_loss + reward_loss + β * kl_loss

        Domain-specific training:
        - MolDreamer:     (MD frame pairs) → coordinate reconstruction + property MSE
        - ProteinDreamer: (sequence, mutation, fitness) → fitness MSE + sequence CE
        - CellDreamer:    (expression, perturbation, outcome) → ZINB NLL + KL

Design notes:
    - KL balancing (DreamerV3): balance the KL between encoder posterior and
      dynamics prior to prevent posterior collapse without restricting the prior.
    - Symlog predictions (DreamerV3): apply symlog transform to fitness targets
      to handle wide dynamic ranges in DMS data.
    - Curriculum: start with short rollouts (1-step prediction), then extend
      to multi-step imagination for compounding-error robustness.
"""
