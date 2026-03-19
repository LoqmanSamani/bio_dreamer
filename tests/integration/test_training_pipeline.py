"""
test_training_pipeline.py — Integration Tests for Training Pipelines.

Purpose:
    End-to-end tests for the full training loop: data loading → model init →
    training → checkpointing → metric logging. Uses small synthetic datasets
    and a few training steps.

Tests to implement:
    - test_world_model_training_loop: WorldModelTrainer runs 3 steps without error,
      loss decreases, checkpoint is saved
    - test_policy_training_loop: PolicyTrainer runs 3 imagination-based updates
    - test_active_learning_loop: one cycle of dream→propose→(mock)evaluate→update
    - test_distributed_training_setup: DDP initialisation with 1 GPU (or CPU fallback)
    - test_mixed_precision: autocast + GradScaler don't produce NaN
    - test_wandb_logging: WandbLogger records metrics (mocked wandb)
    - test_checkpoint_resume: training resumes from checkpoint with correct step count
"""
