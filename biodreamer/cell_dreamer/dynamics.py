"""
biodreamer.cell_dreamer.dynamics — Perturbation-Conditioned Cell Dynamics Model.

Purpose:
    Predicts how a cell's transcriptomic state evolves over time in response
    to perturbations (gene knockouts, drug treatments). This is the "world model"
    that enables planning multi-step intervention strategies in imagination.

Components to implement:
    - CellDynamics(BaseDynamics):
        Perturbation-conditioned latent dynamics:

        1. Neural ODE:
           dz/dt = f_θ(z, p, t) where p = perturbation embedding
           Solved with torchdiffeq (dopri5 or adaptive solvers).
           Produces continuous cell state trajectories.

        2. Neural SDE:
           dz = f_θ(z, p, t)dt + g_θ(z, t)dW
           Adds stochastic noise (biological variability, technical noise).
           Produces a distribution over trajectories.

        3. Conditional Diffusion:
           Generates z_{t+Δt} from z_t conditioned on perturbation p.
           Better for multi-modal outcomes (e.g., a perturbation that causes
           either apoptosis or senescence depending on context).

    - PerturbationEmbedding:
        Encodes perturbation actions into vectors:
        - Gene knockout: one-hot gene ID → embedding (or gene2vec / scGPT embedding)
        - Drug treatment: compound fingerprint + dose + timing → embedding
        - Combinatorial: sum / attention over individual perturbation embeddings

Design notes:
    - The dynamics model must handle variable time gaps (some datasets have
      measurements at t=0, 24h, 72h — irregularly spaced).
    - Combinatorial perturbation prediction is the key challenge: most training
      data covers single perturbations, but the model needs to generalise to
      multi-gene knockouts and drug combos.
"""
