"""
biodreamer.training.schedulers — Learning Rate and KL Schedulers.

Purpose:
    Custom learning rate schedules and KL-weight annealing schedules used
    during world model and policy training.

Components to implement:
    - WarmupCosineScheduler: linear warmup → cosine decay (standard for transformers)
    - CyclicKLScheduler: cyclical KL annealing for VAE training (prevents posterior collapse)
    - LinearWarmupScheduler: simple linear warmup for the first N steps
    - KLBalanceScheduler: DreamerV3-style KL balancing weight schedule

Design notes:
    - Schedulers follow the PyTorch LRScheduler interface where applicable.
    - KL schedulers are separate from LR schedulers — they control the β weight
      in the ELBO loss (β-VAE style) for encoder training.
"""
