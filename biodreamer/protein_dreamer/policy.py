from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch 
import torch.nn as nn
import torch.nn.functional as F
from biodreamer.core.policy import BasePolicy




class ProteinPolicy(BasePolicy):
    """Policy network for mutation planning in ProteinDreamer."""
    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128, 
        policy_model: Optional[Any] = None, 
        loss_fn: Optional[Any] = None,
        device: Optional[torch.device] = None
        ) -> None:
        super().__init__(latent_dim, action_dim)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_model = policy_model.to(self.device) if policy_model is not None else nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)  # Output logits for each possible mutation
        ).to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    
    def select_action(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select the mutation action with the highest predicted value."""
        logits = self.get_action_distribution(z_t)
        action = torch.argmax(logits, dim=-1)  # Shape: (batch_size,)
        return action
    
    def get_action_distribution(self, z_t: torch.Tensor) -> torch.Tensor:
        """Get the logits for each possible mutation action."""
        return self.policy_model(z_t)  # Shape: (batch_size, action_dim)
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Update the policy parameters based on a batch of experience."""
        # Placeholder implementation — actual loss computation depends on the RL algorithm used (e.g., PPO, DQN)
        states = batch['states']  # Shape: (batch_size, latent_dim)
        actions = batch['actions']  # Shape: (batch_size,)
        rewards = batch['rewards']  # Shape: (batch_size,)
        
        logits = self.get_action_distribution(states)  # Shape: (batch_size, action_dim)
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, action_dim)
        
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]  # Shape: (batch_size,)
    
        policy_loss = -self.loss_fn(selected_log_probs, rewards)
        
        return {'policy_loss': policy_loss}
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action logits for a given latent state."""
        return self.get_action_distribution(z_t)









