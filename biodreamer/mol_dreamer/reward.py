"""
biodreamer.mol_dreamer.reward — Molecular Property Predictors.

Purpose:
    Predicts biophysical properties from latent molecular states that serve
    as reward signals for RL-based molecular optimisation.

Components to implement:
    - MolRewardHead(BaseRewardHead):
        - predict_binding_affinity(z_t) → ΔG (kcal/mol)
        - predict_stability(z_t) → Tm or ΔΔG
        - predict_sasa(z_t) → solvent-accessible surface area
        - predict_multi(z_t) → dict of all properties

    Training data sources:
        - PDBBind (binding affinities for protein-ligand complexes)
        - Thermal stability measurements from literature
        - SASA computed from MD trajectories (self-supervised signal)

Design notes:
    - Multi-task training on all property heads simultaneously, with task-specific
      loss weights configured in configs/mol_dreamer/default.yaml.
    - Uncertainty estimation (ensemble or evidential) for each property,
      required by the active inference module.
"""
