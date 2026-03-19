"""
test_dynamics.py — Unit Tests for biodreamer.core.dynamics.

Tests to implement:
    - test_dynamics_transition: given (z_t, action) → z_{t+1}
    - test_dynamics_distribution: output includes mean and std for stochastic transitions
    - test_dynamics_deterministic_mode: toggle deterministic flag
    - test_dynamics_sequence: multi-step unrolling produces valid trajectories
"""
