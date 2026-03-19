"""
biodreamer.cell_dreamer — CellDreamer: World Models for Gene Regulatory Dynamics.

This sub-package learns world models of cellular dynamics from perturbation
data (Perturb-seq, CRISPRi/a, drug screens) and uses RL to plan optimal
multi-step intervention strategies for cell fate control.

Modules:
    - encoder:     scRNA-seq VAE encoder (expression profiles → latent cell states)
    - dynamics:    Neural ODE/SDE for perturbation-conditioned cell state evolution
    - decoder:     Gene expression profile reconstruction
    - reward:      Cell state objective (distance to target phenotype)
    - policy:      Perturbation sequence planner (RL agent)
    - environment: Gene regulatory network environment wrapper
    - data/        Perturb-seq and scRNA-seq data loading
"""
