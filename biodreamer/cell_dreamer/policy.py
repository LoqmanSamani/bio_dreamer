"""
biodreamer.cell_dreamer.policy — Perturbation Sequence Planner.

Purpose:
    RL agent that plans optimal multi-step perturbation strategies (which genes
    to knock out/activate, which drugs to apply, in what order) to drive cells
    toward a desired phenotype.

Components to implement:
    - CellPolicy(BasePolicy):
        - Action space: (perturbation_type × target_gene × dose × timing)
          This is a large combinatorial space — the policy must handle it
          efficiently (e.g., factored action representation).

    Policy types:
        1. CellPPO: PPO with factored action heads for each dimension.
        2. CellMCTS: Tree search over perturbation sequences using the
           world model for rollouts.
        3. CellActiveInference: EFE-minimising policy for exploration-aware
           perturbation planning.

Design notes:
    - Perturbation order matters: knocking out gene A then B may have a
      different outcome than B then A (non-commutative interventions).
    - The policy should support constraints (e.g., "don't knock out essential
      genes", "maximum 3 perturbations per round").
    - Budget-constrained planning: real experiments have cost — the policy
      should maximise reward within a perturbation budget.
"""
