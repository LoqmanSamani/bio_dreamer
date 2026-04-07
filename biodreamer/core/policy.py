from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn




class BasePolicy(ABC, nn.Module):
    """Abstract base class for all RL policies in BioDreamer.

    Subclasses must implement:
        - select_action(z_t) → action tensor
        - update(batch) → dict of loss values
    """

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._action_dim = action_dim

    @abstractmethod
    def select_action(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select the best action given the current latent state.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            Action tensor (format depends on domain).
        """
        ...

    def select_action_with_exploration(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select an action with added exploration noise / entropy.

        Default implementation delegates to select_action().
        Subclasses should override to add stochasticity for training.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            Action tensor with exploration noise.
        """
        return self.select_action(z_t)

    def get_action_distribution(self, z_t: torch.Tensor) -> Any:
        """Return a distribution over actions for the given latent state.

        Optional — subclasses that maintain an explicit policy distribution
        (e.g. PPO, SAC) should override this.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            A torch.distributions.Distribution or equivalent.
        """
        raise NotImplementedError(
            "get_action_distribution() is optional; override in subclass."
        )

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy parameters given a batch of experience.

        Args:
            batch: Dict with keys like 'z_t', 'actions', 'rewards',
                'next_z_t', etc.  Exact keys depend on the algorithm.

        Returns:
            Dict of scalar loss / metric values for logging.
        """
        ...

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """nn.Module forward pass — delegates to select_action()."""
        return self.select_action(z_t)
