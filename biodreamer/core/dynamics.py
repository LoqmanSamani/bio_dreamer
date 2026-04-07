"""
biodreamer.core.dynamics — Base dynamics model interface.

Purpose:
    Abstract base class for latent dynamics models. A dynamics model predicts
    how the latent state evolves after an action is taken:
        z_{t+1} = f_theta(z_t, a_t)

    This is the "world model" in the narrow sense — it learns the transition
    function of the biological environment in latent space.

Components to implement:
    - BaseDynamics(nn.Module):
        - predict(z_t, action) → z_{t+1} (next latent state)
        - predict_distribution(z_t, action) → distribution over z_{t+1}
          (for uncertainty estimation and active inference)
        - rollout(z_0, actions_sequence) → list of z_t states

    Domain-specific dynamics models:
        - MolDreamer:     Latent diffusion / neural SDE for molecular coordinate evolution
        - ProteinDreamer: Conditional transformer or latent diffusion predicting
                          structure+fitness shifts after mutations
        - CellDreamer:    Neural ODE/SDE for temporal gene regulatory dynamics

Design notes:
    - Support both deterministic and stochastic transitions (RSSM-style).
    - The stochastic component is critical for active inference: the agent
      needs to estimate uncertainty over next states to compute epistemic value.
    - Support variable time steps (especially for CellDreamer Neural ODE and
      MolDreamer coarse-grained jumps).
      
      
      
BaseDynamics(nn.Module):
    @abstractmethod predict(z_t, action_emb) → ẑ_{t+1}
    predict_distribution(z_t, action_emb, n_samples) → {ẑ_{t+1}^(n)}
    rollout(z_0, actions, horizon) → [ẑ_1, ..., ẑ_H]
"""

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
    
