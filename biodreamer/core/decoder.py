from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn





class BaseDecoder(ABC, nn.Module):
    """Abstract base class for all domain-specific decoders in BioDreamer.

    Subclasses must implement:
        - decode(z_t) → reconstructed observation
        - decode_batch(z_batch) → batch of reconstructed observations
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        
    @abstractmethod
    def decode(self, z_t: torch.Tensor) -> Any:
        """Decode a latent state z_t into a reconstructed observation.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Reconstructed observation in domain-specific format.
        """
        ...
    
    
    def decode_batch(self, z_batch: torch.Tensor) -> Any:
        """Decode a batch of latent states into reconstructed observations.

        Default implementation applies decode() to the batch.
        Subclasses may override for more efficient batched decoding.

        Args:
            z_batch: Batch of latent state tensors of shape (B, ..., latent_dim).

        Returns:
            Batch of reconstructed observations in domain-specific format.
        """
        return self.decode(z_batch)

    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self._latent_dim

    def forward(self, z_t: torch.Tensor) -> Any:
        """nn.Module forward pass — delegates to decode()."""
        return self.decode(z_t)
    
