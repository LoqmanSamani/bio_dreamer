"""
train_policy.py — Policy Training Script (Imagination-Based).

Purpose:
    CLI entry point for training the actor-critic policy inside a
    trained world model. The policy learns by "dreaming" in latent
    space — no real environment interaction needed.

Usage:
    python scripts/train_policy.py \\
        --module protein_dreamer \\
        --config configs/protein_dreamer/default.yaml \\
        --world-model checkpoints/protein_dreamer/world_model.pt \\
        --output-dir checkpoints/protein_dreamer/policy/ \\
        [--policy-type PPO] \\
        [--dreaming-horizon 15] \\
        [--wandb-project biodreamer]

Arguments:
    --module:           Module name
    --config:           YAML config path
    --world-model:      Pre-trained world model checkpoint
    --output-dir:       Policy checkpoint output directory
    --policy-type:      PPO | SAC | MCTS | ActiveInference
    --dreaming-horizon: Number of imagination steps per episode
    --wandb-project:    W&B project name

Workflow:
    1. Load config and world model checkpoint (freeze world model weights)
    2. Instantiate policy network (module-specific)
    3. Create PolicyTrainer with imagination-based rollouts
    4. Train: at each step, dream H-step trajectories, compute returns,
       update actor and critic via chosen algorithm
    5. Save policy checkpoint
"""
