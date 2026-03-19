"""
train_world_model.py — World Model Training Script.

Purpose:
    CLI entry point for training a world model for any BioDreamer module.
    Loads configuration, initialises data loaders, model, and trainer,
    then runs the training loop.

Usage:
    python scripts/train_world_model.py \\
        --module protein_dreamer \\
        --config configs/protein_dreamer/default.yaml \\
        --output-dir checkpoints/protein_dreamer/ \\
        [--resume-from checkpoints/protein_dreamer/latest.pt] \\
        [--wandb-project biodreamer] \\
        [--gpus 0,1] \\
        [--seed 42]

Arguments:
    --module:       One of {mol_dreamer, protein_dreamer, cell_dreamer}
    --config:       Path to YAML config file
    --output-dir:   Directory for checkpoints and logs
    --resume-from:  Optional checkpoint path for resuming training
    --wandb-project: W&B project name (None to disable logging)
    --gpus:         Comma-separated GPU indices
    --seed:         Random seed for reproducibility

Workflow:
    1. Parse CLI args and load YAML config (merge overrides)
    2. Set random seeds (torch, numpy, python)
    3. Initialise dataset and DataLoader
    4. Instantiate module-specific encoder, dynamics, decoder, reward
    5. Compose into WorldModel
    6. Create WorldModelTrainer with callbacks
    7. Call trainer.fit()
    8. Save final checkpoint and log summary
"""
