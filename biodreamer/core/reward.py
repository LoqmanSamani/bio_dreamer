from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn





class BaseRewardHead(ABC, nn.Module):
    """abstract base class for all domain-specific reward heads in BioDreamer.

    subclasses must implement:
        - predict(z_t) → reward (scalar tensor)
        - predict_multi(z_t) → dict of named rewards (for multi-objective)
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        
    @abstractmethod
    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        """predict a scalar reward from a latent state z_t.

        args:
            z_t: Latent state tensor of shape (..., latent_dim).

        returns:
            scalar reward tensor of shape (...,).
        """
        ...
    
    
    def predict_multi(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """predict per-objective rewards from a latent state z_t.

        default implementation returns the scalar prediction under a
        generic key.  domain-specific subclasses should override this to
        return a dict of named reward components (e.g. stability,
        affinity, activity).

        args:
            z_t: Latent state tensor of shape (..., latent_dim).

        returns:
            dict mapping objective names to reward tensors.
        """
        return {"reward": self.predict(z_t)}

    def get_latent_dim(self) -> int:
        """return the dimensionality of the latent representation."""
        return self._latent_dim

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """forward pass, delegates to predict()."""
        return self.predict(z_t)