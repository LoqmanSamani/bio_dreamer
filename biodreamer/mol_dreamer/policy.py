"""
biodreamer.mol_dreamer.policy — RL Policy for Molecular Optimisation.

Purpose:
    RL agent that operates within the MolDreamer world model to propose
    molecular modifications (mutations, ligand edits, simulation parameters)
    that optimise target properties.

Components to implement:
    - MolPolicy(BasePolicy):
        Continuous-action actor-critic (SAC) operating in latent action space.
        Actions are decoded into:
            - Single-residue mutations (for protein MD optimisation)
            - Ligand atom/group modifications (for drug design)
            - Simulation parameter changes (temperature, pressure, restraints)

Design notes:
    - Trained entirely in imagination (world model rollouts), not real MD.
    - The policy proposes a sequence of modifications and the world model
      simulates the resulting trajectory, avoiding costly full MD runs.
    - Can optionally integrate the ActiveInferencePolicy from core for
      exploration-driven molecular design.
"""
