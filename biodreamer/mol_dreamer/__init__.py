"""
biodreamer.mol_dreamer — MolDreamer: World Models for Molecular Dynamics.

This sub-package implements the MolDreamer module, which learns world models
of molecular dynamics trajectories and uses model-based RL to optimise
molecular properties (binding affinity, stability, solvent accessibility).

Modules:
    - encoder:       SE(3)-equivariant GNN encoding atomic systems → latent states
    - dynamics:      Latent diffusion / neural SDE for molecular state evolution
    - decoder:       Reconstructs 3D coordinates and observables from latent states
    - reward:        Property predictors (ΔG, Tm, SASA)
    - policy:        Actor-critic for molecular optimisation in latent space
    - environment:   MD environment wrapper (interfaces with OpenMM / GROMACS)
    - data/          Data loading and preprocessing for MD trajectories
"""
