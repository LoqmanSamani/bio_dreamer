from abc import ABC, abstractmethod
from typing import Any, Dict

import torch 
import torch.nn as nn





class BasePolicy(ABC, nn.Module):
    """Abstract base class for mutation planning policies in ProteinDreamer.
    
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
    
    @abstractmethod
    def select_action(self, z_t: torch.Tensor) -> Any:
        """Select a mutation action given the current latent state z_t.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Action in domain-specific format (e.g., (position, amino_acid, edit_type)).
        """
        ...
        
    def select_action_with_exploration(self, z_t: torch.Tensor) -> Any:
        """Select an action with exploration (e.g., epsilon-greedy, sampling from distribution).

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Action in domain-specific format.
        """
        # Default implementation: just call select_action (no exploration)
        return self.select_action(z_t)
    
    def get_action_distribution(self, z_t: torch.Tensor) -> Any:
        """Get the action distribution (e.g., logits or probabilities) for a given latent state.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).
        Returns:
            Action distribution in domain-specific format (e.g., logits for each possible mutation).
        """
        # Default implementation: return None (not all policies need this)
        return None
      
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Update the policy parameters based on a batch of experience.

        Args:
            batch: Dictionary containing training data (e.g., states, actions, rewards, next_states).

        Returns:
            Dictionary of loss values for logging (e.g., {'policy_loss': ..., 'value_loss': ...}).
        """
        ...
        
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self._latent_dim
    
    def forward(self, z_t: torch.Tensor) -> Any:
        """nn.Module forward pass — delegates to select_action()."""
        return self.select_action(z_t)
    