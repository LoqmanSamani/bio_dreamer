from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn





class BaseRewardHead(ABC, nn.Module):
    """Abstract base class for all domain-specific reward heads in BioDreamer.

    Subclasses must implement:
        - predict(z_t) → reward (scalar tensor)
        - predict_multi(z_t) → dict of named rewards (for multi-objective)
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        
    @abstractmethod
    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict a scalar reward from a latent state z_t.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Scalar reward tensor of shape (...,).
        """
        ...
    
    
    def predict_multi(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict per-objective rewards from a latent state z_t.

        Default implementation returns the scalar prediction under a
        generic key.  Domain-specific subclasses should override this to
        return a dict of named reward components (e.g. stability,
        affinity, activity).

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Dict mapping objective names to reward tensors.
        """
        return {"reward": self.predict(z_t)}

    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self._latent_dim

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """nn.Module forward pass — delegates to predict()."""
        return self.predict(z_t)

