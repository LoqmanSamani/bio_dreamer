from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn





class BaseEncoder(ABC, nn.Module):
    """Abstract base class for all domain-specific encoders in BioDreamer.

    Subclasses must implement:
        - encode(observation) → z_t tensor of shape (..., latent_dim)
        - get_latent_dim() → int
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim

    @abstractmethod
    def encode(self, observation: Any) -> torch.Tensor:
        """Encode an observation into a latent representation z_t.

        Args:
            observation: Domain-specific input (e.g. token IDs, graphs,
                expression vectors). Format defined by each subclass.

        Returns:
            Latent state tensor of shape (..., latent_dim).
        """
        ...

    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self._latent_dim

    def forward(self, observation: Any) -> torch.Tensor:
        """nn.Module forward pass — delegates to encode()."""
        return self.encode(observation)

    def freeze(self) -> None:
        """Freeze all parameters (useful for pre-trained backbones)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
