"""
biodreamer.core.policy — Base RL policy interface.

Purpose:
    Abstract base class for RL policies that select actions (mutations,
    perturbations, parameter changes) to optimise rewards within the
    learned world model.

Components to implement:
    - BasePolicy(nn.Module):
        - select_action(z_t) → action
        - select_action_with_exploration(z_t) → action (with entropy/noise)
        - get_action_distribution(z_t) → distribution over actions
        - update(trajectories) → loss dict (for on-policy methods)

    Concrete policy types to support:
        - LatentActorCritic:  SAC / PPO operating in continuous latent action space
        - DiscreteMutationPolicy: PPO / DQN for discrete mutation actions
                                  (which position × which amino acid)
        - MCTSPlanner: Monte Carlo Tree Search over mutation trees using
                       the world model for rollout evaluation
        - GFlowNetPolicy: Diversity-seeking alternative that samples
                          proportionally to reward

Design notes:
    - Policies are trained entirely "in imagination" — they act in the world
      model, not in the real environment. This is the key advantage of MBRL.
    - The MCTS planner uses the dynamics model for forward simulation and
      the reward head for leaf evaluation — no separate value network needed.
    - For ProteinDreamer, the action space is a structured discrete space:
      (position ∈ {1..L}, amino_acid ∈ {A..Y}, edit_type ∈ {sub, ins, del}).
"""
