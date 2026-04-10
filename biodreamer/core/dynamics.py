from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn




class BaseDynamics(ABC, nn.Module):
    """Abstract base class for all domain-specific dynamics models in BioDreamer.

    Subclasses must implement:
        - predict(z_t, action) → z_{t+1} (next latent state)
        - predict_distribution(z_t, action) → distribution over z_{t+1}
          (for uncertainty estimation and active inference)
        - rollout(z_0, actions_sequence) → list of z_t states
    """
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state z_{t+1} given the current state and action embedding.

        Args:
            z_t (torch.Tensor): Current latent state tensor of shape (..., latent_dim).
            action_emb (torch.Tensor): Action embedding tensor of shape (..., action_dim).

        Returns:
            torch.Tensor: Predicted next latent state tensor of shape (..., latent_dim).
        """
        ...
        
    def predict_distribution(self, z_t: torch.Tensor, action_emb: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Predict a distribution over next latent states z_{t+1} for uncertainty estimation.

        Args:
            z_t (torch.Tensor): Current latent state tensor of shape (..., latent_dim).
            action_emb (torch.Tensor): Action embedding tensor of shape (..., action_dim).
            n_samples (int): Number of samples to draw from the predicted distribution. 
        """
        raise NotImplementedError("predict_distribution() is optional and may not be implemented by all dynamics models.")
    
    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor, horizon: int) -> torch.Tensor:
        """Rollout a sequence of latent states given an initial state and a sequence of actions.

        Args:
            z_0 (torch.Tensor): Initial latent state tensor of shape (..., latent_dim).
            actions (torch.Tensor): Sequence of action embeddings of shape (..., horizon, action_dim).
            horizon (int): Number of time steps to rollout.
        Returns:
            torch.Tensor: Sequence of predicted latent states of shape (..., horizon, latent_dim).
        """
        z_t = z_0
        predicted_states = []
        for t in range(horizon):
            action_t = actions[:, t, :]
            z_t = self.predict(z_t, action_t)
            predicted_states.append(z_t)
        return torch.stack(predicted_states, dim=1)
    
    
    def forward(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Alias for predict() to allow calling the dynamics model directly."""
        return self.predict(z_t, action_emb)
    
