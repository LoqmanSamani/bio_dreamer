"""
biodreamer.training — Training loops for world models and RL policies.

Modules:
    - trainer:              Base trainer class with PyTorch DDP support
    - world_model_trainer:  Training loop for the encoder+dynamics+decoder+reward
    - policy_trainer:       RL policy training in imagination (dreaming)
    - active_learning:      Active learning loop (propose → evaluate → retrain)
    - callbacks:            Training callbacks (logging, checkpointing, early stopping)
    - schedulers:           Learning rate and KL-balancing schedulers
"""
