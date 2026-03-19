"""
biodreamer.training.trainer — Base Trainer Class.

Purpose:
    Provides the foundational training infrastructure shared by all BioDreamer
    training loops: device management, distributed training (DDP), mixed precision,
    checkpointing, logging, and callback hooks.

Components to implement:
    - BaseTrainer:
        - __init__(model, config, device)
        - train(train_loader, val_loader, n_epochs) → training history
        - validate(val_loader) → validation metrics dict
        - save_checkpoint(path)
        - load_checkpoint(path)
        - setup_distributed() — initialise PyTorch DDP if multiple GPUs
        - setup_mixed_precision() — configure torch.amp / bfloat16

    Configuration:
        - Loaded from YAML config files (configs/<module>/default.yaml)
        - Overridable via CLI arguments

Design notes:
    - Built on pure PyTorch (no Lightning or other wrappers) for maximum
      control and transparency — important for a research project.
    - Supports Weights & Biases (wandb) logging out of the box.
    - Checkpoint format: dict with model_state_dict, optimizer_state_dict,
      epoch, best_metric, config — compatible with HF Hub upload.
"""
