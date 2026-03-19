"""
biodreamer.cell_dreamer.environment — Gene Regulatory Network Environment.

Purpose:
    Wraps perturbation datasets and in-silico GRN simulators as a gym-like
    environment for RL training and evaluation.

Components to implement:
    - CellEnvironment:
        - reset(cell_type, initial_expression) → initial cell state
        - step(perturbation) → (next_state, reward, done, info)
        - render() → UMAP coordinates + expression heatmap data

    Environment variants:
        1. PerturbSeqReplayEnvironment: replays real Perturb-seq data.
           Used for world model training (ground-truth transitions).
        2. BooleanNetworkEnvironment: uses a Boolean GRN model as ground truth.
           Good for controlled experiments (known dynamics, known optimum).
        3. InSilicoGRN: uses SERGIO or BoolODE to simulate stochastic GRN dynamics.
           Provides a synthetic but biologically plausible environment.

Design notes:
    - Single-cell data is inherently stochastic (each cell is different).
      The environment should provide population-level statistics (mean,
      variance over cells) rather than single-cell observations.
    - Time-course observations: the environment can return trajectories
      (not just next-step observations) for Neural ODE training.
"""
