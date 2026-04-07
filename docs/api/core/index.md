# Core Module Overview

The `biodreamer.core` package provides abstract base classes that all three
domain modules (ProteinDreamer, MolDreamer, CellDreamer) inherit from.

| Class | File | Role |
|---|---|---|
| [`BaseEncoder`](encoder.md) | `core/encoder.py` | Observation → latent state $z_t$ |
| [`BaseDynamics`](dynamics.md) | `core/dynamics.py` | $(z_t, a_t) \to z_{t+1}$ transition model |
| [`BaseDecoder`](decoder.md) | `core/decoder.py` | $z_t$ → reconstructed observation |
| [`BaseRewardHead`](reward.md) | `core/reward.py` | $z_t$ → scalar fitness reward |
| [`BasePolicy`](policy.md) | `core/policy.py` | RL policy for action selection |
| [`ActiveInference`](active_inference.md) | `core/active_inference.py` | Expected Free Energy & policy |
| [`WorldModel`](world_model.md) | `core/world_model.py` | Encoder + Dynamics + Reward composite |
