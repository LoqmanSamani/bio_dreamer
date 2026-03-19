"""
biodreamer.core — Shared base classes and abstractions.

This sub-package defines the domain-agnostic skeleton that all three BioDreamer
modules (MolDreamer, ProteinDreamer, CellDreamer) inherit from:
    - WorldModel      — composite container of encoder + dynamics + decoder + reward
    - BaseEncoder     — maps raw observations to latent states
    - BaseDynamics    — predicts latent state transitions given actions
    - BaseDecoder     — reconstructs observables from latent states
    - BaseRewardHead  — predicts scalar rewards from latent states
    - BasePolicy      — RL policy that selects actions in latent space
    - ActiveInference — expected free energy computation and policy selection
"""
