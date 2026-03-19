"""
biodreamer.protein_dreamer — ProteinDreamer: Model-Based RL for Protein Design.

The core module of the BioDreamer project. Frames protein design as a Markov
Decision Process where a world model learns the protein fitness landscape and
an RL agent plans multi-step mutation strategies "in imagination."

Modules:
    - encoder:       ESM-2 (sequence) + GVP-GNN (structure) → joint latent state
    - dynamics:      Conditional transformer / latent diffusion for mutation transitions
    - decoder:       Fitness scores + mutated sequence + predicted structure
    - reward:        Multi-objective fitness prediction (ΔΔG, Kd, kcat)
    - policy:        SAC/PPO + MCTS mutation planner
    - environment:   Protein fitness landscape environment (gym-like interface)
    - uncertainty:   Epistemic uncertainty module (ensemble / evidential DL)
    - data/          ProteinGym, Tsuboyama mega-scale data loading
"""
