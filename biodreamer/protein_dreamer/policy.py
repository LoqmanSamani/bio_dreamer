"""
biodreamer.protein_dreamer.policy — Mutation Planning Policy.

Purpose:
    RL agent that selects amino acid mutations to optimise protein fitness.
    Operates entirely within the learned world model ("dreaming") — plans
    multi-step mutation trajectories before proposing candidates for evaluation.

Components to implement:
    - ProteinPolicy(BasePolicy):
        Discrete-action RL policy for protein mutations:
        - Action space: (position ∈ {1..L}, amino_acid ∈ {A..Y}, edit_type ∈ {sub, ins, del})
        - Encodes action as (positional embedding + AA embedding + edit embedding)

    Concrete policy implementations:
        1. ProteinPPO: Proximal Policy Optimisation in latent space.
           Standard choice for discrete action spaces.

        2. ProteinSAC: Soft Actor-Critic with Gumbel-Softmax for discrete actions.
           Better exploration via maximum entropy objective.

        3. ProteinMCTS: Monte Carlo Tree Search over mutation trees.
           Uses the world model as the forward simulator and reward head for
           leaf evaluation. Most aligned with the "planning in imagination" vision.
           Inspired by EvoPlay (Wang et al., 2023) but uses a full world model
           instead of a discriminative fitness oracle.

        4. ProteinActiveInferencePolicy: Wraps ActiveInferencePolicy from core.
           Selects mutations by minimising Expected Free Energy, balancing
           fitness optimisation (pragmatic value) with uncertainty reduction
           (epistemic value).

Design notes:
    - The policy proposes a full mutation path (sequence of T mutations), not
      just a single mutation. The world model evaluates the full path.
    - Candidate filtering: after dreaming N trajectories, rank by predicted
      fitness and diversity (sequence + structure diversity in latent space).
    - Curriculum: start with single-mutation planning (T=1), then scale to
      multi-step trajectories (T=3→5→10) as the world model improves.
"""
