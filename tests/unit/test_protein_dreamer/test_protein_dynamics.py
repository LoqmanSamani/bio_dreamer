"""
test_protein_dynamics.py — Unit Tests for biodreamer.protein_dreamer.dynamics.

Tests to implement:
    - test_mutation_conditioned_transition: mutation embedding modifies transition
    - test_epistasis_modelling: multi-mutation interactions captured
    - test_latent_diffusion_mode: diffusion-based dynamics produces valid distributions
    - test_rssm_mode: RSSM variant maintains recurrent state correctly
    - test_uncertainty_propagation: uncertainty grows with rollout length
"""
