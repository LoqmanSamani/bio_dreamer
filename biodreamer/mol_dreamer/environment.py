"""
biodreamer.mol_dreamer.environment — Molecular Dynamics Environment Wrapper.

Purpose:
    Wraps molecular dynamics engines (OpenMM, GROMACS) and trajectory datasets
    into a gym-like environment interface for RL training and evaluation.

Components to implement:
    - MolEnvironment:
        - reset(pdb_path) → initial observation (atomic coords, features)
        - step(action) → (next_observation, reward, done, info)
        - render() → 3D coordinates for visualisation

    - TrajectoryReplayEnvironment:
        Replays pre-computed MD trajectories as if they were live environments.
        Used for world model training (the environment provides ground-truth
        transitions) and for offline RL.

    - OracleEnvironment:
        Runs actual MD simulation steps using OpenMM as ground truth.
        Used for validation and active learning (expensive, used sparingly).

Design notes:
    - The environment abstraction follows the OpenAI Gym / Gymnasium interface.
    - Most training happens against the world model (MolDynamics), not this
      environment. This module is for data collection and final validation.
"""
