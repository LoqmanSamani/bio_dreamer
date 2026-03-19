"""
biodreamer.inference.dreamer — Imagination Rollout Engine.

Purpose:
    The "dreaming" engine — generates imagined trajectories in the world model
    by repeatedly applying the policy and dynamics model:
        z_0 → (a_0) → z_1 → (a_1) → z_2 → ... → z_T

    This is the core of model-based RL: planning in imagination before proposing
    real-world candidates.

Components to implement:
    - Dreamer:
        - dream(world_model, policy, z_0, n_trajectories, horizon) → DreamResult
        - dream_batch(z_0_batch, ...) → batched rollouts (GPU-efficient)
        - dream_with_branching(z_0, branch_factor, depth) → tree of trajectories

    - DreamResult:
        - latent_trajectories: tensor (n_traj, horizon, latent_dim)
        - actions: tensor (n_traj, horizon, action_dim)
        - predicted_rewards: tensor (n_traj, horizon)
        - uncertainties: tensor (n_traj, horizon)
        - cumulative_rewards: tensor (n_traj,)

Design notes:
    - Dreaming should be parallelised across trajectories on GPU.
    - For MCTS-based policies, dreaming IS the planning — the tree search
      uses the world model for forward simulation at each node.
    - Support early termination: stop a trajectory if reward drops below
      threshold or uncertainty exceeds a bound.
    - Trajectory diversity: optionally enforce diversity among the N dreamed
      trajectories (e.g., different first mutations) to avoid mode collapse.
"""
