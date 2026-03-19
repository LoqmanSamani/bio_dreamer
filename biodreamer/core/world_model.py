"""
biodreamer.core.world_model — Base WorldModel class.

Purpose:
    Defines the abstract WorldModel that composes an encoder, dynamics model,
    decoder, and reward head into a single module. This is the central
    abstraction of BioDreamer — all three domain modules instantiate this
    class with their own domain-specific components.

Components to implement:
    - WorldModel(nn.Module):
        - self.encoder:  BaseEncoder   — observation → latent state z_t
        - self.dynamics: BaseDynamics  — (z_t, action) → z_{t+1}
        - self.decoder:  BaseDecoder   — z_t → reconstructed observation
        - self.reward:   BaseRewardHead — z_t → scalar reward
        - imagine(z_0, actions) → full trajectory of latent states + predicted rewards
        - encode(observation) → z_t
        - decode(z_t) → observation
        - predict_reward(z_t) → reward

    This class is domain-agnostic. MolDreamer, ProteinDreamer, and CellDreamer
    each provide concrete encoder/dynamics/decoder/reward implementations and
    pass them to WorldModel.

Design notes:
    - Follow DreamerV3's architecture: RSSM with deterministic + stochastic states.
    - Support both continuous latent spaces (diffusion dynamics) and discrete
      token spaces (IRIS-style autoregressive dynamics).
    - The `imagine` method must support variable-length rollouts with optional
      early stopping based on reward thresholds or uncertainty bounds.
"""
