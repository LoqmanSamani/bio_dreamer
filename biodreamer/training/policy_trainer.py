"""
biodreamer.training.policy_trainer — RL Policy Training in Imagination.

Purpose:
    Trains the RL policy (actor-critic, MCTS, or active inference) entirely
    within the learned world model. The policy "dreams" — generates imagined
    trajectories in the world model and optimises its behaviour on them.

Components to implement:
    - PolicyTrainer(BaseTrainer):
        Training procedure (DreamerV3 actor-critic):
        1. Sample initial latent states z_0 from the replay buffer (real data)
        2. Imagine trajectories: z_0 → z_1 → ... → z_H using policy + dynamics
        3. Compute imagined rewards r_1, ..., r_H from reward head
        4. Compute value targets (lambda-returns or GAE)
        5. Update actor (policy gradient) and critic (value regression)

        For Active Inference variant:
        1-2. Same as above + compute epistemic value (uncertainty reduction)
        3. Compute Expected Free Energy = pragmatic_value + epistemic_value
        4. Update policy to minimise EFE

    Policy types supported:
        - ActorCritic (PPO / SAC) — standard RL baseline
        - ActiveInferencePolicy — EFE-minimising (ProteinDreamer's default)
        - MCTS — tree search doesn't need gradient training, but the value
          network and prior policy network do.

Design notes:
    - The policy NEVER interacts with the real environment during training.
      All training is "in imagination" — this is the entire point of MBRL.
    - Imagination horizon H is a key hyperparameter: too short = myopic,
      too long = compounding model errors degrade signal.
    - Exploration: the policy should be encouraged to visit uncertain regions
      of the world model (via entropy bonus or active inference epistemic term).
"""
