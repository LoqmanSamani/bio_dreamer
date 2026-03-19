"""
test_world_model.py — Unit Tests for biodreamer.core.world_model.

Purpose:
    Validates the base WorldModel class: initialisation, forward pass,
    imagine() rollout, and component composition.

Tests to implement:
    - test_world_model_init: components (encoder, dynamics, decoder, reward)
      are properly composed and accessible
    - test_forward_pass: given observation + action, returns next latent,
      reconstruction, and reward
    - test_imagine_rollout: imagine(horizon=H) produces H-step trajectory
      of latent states, actions, rewards
    - test_imagine_shapes: output tensor shapes match expected dimensions
    - test_imagine_with_policy: policy generates actions for each step
    - test_kl_divergence_computation: posterior vs. prior KL is non-negative
    - test_checkpoint_save_load: save and reload produces identical outputs
"""
