"""
biodreamer.training.callbacks — Training Callbacks.

Purpose:
    Hook-based callback system for injecting logic into the training loop
    at specific points (epoch start/end, batch start/end, validation end).

Components to implement:
    - Callback (base class):
        - on_train_start(trainer)
        - on_epoch_start(trainer, epoch)
        - on_batch_end(trainer, batch_idx, loss_dict)
        - on_epoch_end(trainer, epoch, metrics)
        - on_train_end(trainer)

    Concrete callbacks:
        - WandbLogger: logs metrics, histograms, and images to W&B
        - CheckpointCallback: saves model checkpoints at regular intervals
          and keeps top-K by validation metric
        - EarlyStoppingCallback: stops training if validation metric plateaus
        - LearningRateFinder: auto-finds optimal learning rate (optional)
        - GradientMonitorCallback: logs gradient norms and detects exploding/vanishing
"""
