"""
biodreamer.protein_dreamer.dynamics — Protein Fitness Landscape Dynamics Model.

Purpose:
    The core "world model" for ProteinDreamer. Predicts how the protein's
    latent state (encoding sequence, structure, and fitness) changes after
    a mutation is applied:
        z_{t+1} = f_θ(z_t, mutation_t)

    This allows the agent to "dream" — simulate entire mutation trajectories
    without querying expensive structure predictors or wet-lab assays at each step.

Components to implement:
    - ProteinDynamics(BaseDynamics):
        Mutation-conditioned latent transition model. Options:

        1. Conditional Transformer:
           Input: (z_t, mutation_embedding) → transformer layers → z_{t+1}
           Mutation embedding encodes: (position, wild-type AA, mutant AA, edit_type)

        2. Latent Diffusion:
           Denoising diffusion in latent space conditioned on (z_t, mutation).
           Produces a distribution over z_{t+1}, capturing multi-modal outcomes
           (a mutation can have multiple structural consequences).

        3. RSSM (DreamerV3-style):
           Deterministic path: h_{t+1} = GRU(h_t, z_t, mutation)
           Stochastic state: z_{t+1} ~ q(z | h_{t+1})
           Combines deterministic tracking of mutation history with stochastic
           modelling of structural uncertainty.

    - MutationEmbedding:
        Encodes a mutation action into a fixed-size vector:
        (position_encoding, AA_embedding[wt], AA_embedding[mut], edit_type_embedding)

Design notes:
    - The dynamics model is trained on DMS data: input = (wild-type latent, mutation)
      → target = mutant fitness (and optionally mutant latent from ESMFold structure).
    - Multi-step rollouts accumulate mutations: z_0 → z_1 → ... → z_T.
      The model must handle compounding errors gracefully (symlog predictions,
      latent regularisation, truncated rollout training).
    - Epistasis modelling: the dynamics model implicitly captures epistatic effects
      because it conditions on the current state z_t (which encodes all prior mutations),
      not just the wild-type.
"""
