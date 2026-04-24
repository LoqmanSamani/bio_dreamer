import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from biodreamer.core.dynamics import BaseDynamics




class ProteinDynamics(BaseDynamics):
    """mutation-conditioned latent transition model for ProteinDreamer"""
    def __init__(self, predictor: Any, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.predictor = predictor.to(self.device)
        
    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """predict the next latent state z_{t+1} given current state z_t and action embedding"""
        return self.predict_distribution(z_t, action_emb, n_samples)
    
    
    def predict_distribution(self, z_t: torch.Tensor, action_emb: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:    
        """predict a distribution over next latent states z_{t+1} for uncertainty estimation"""
        z_nexts = [self.predictor(z_t, action_emb) for _ in range(n_samples)]
        z_next_dist = torch.stack(z_nexts, dim=0)  # shape: (n_samples, batch_size, latent_dim)
        return {"mean": torch.mean(z_next_dist, dim=0), "var": torch.var(z_next_dist, dim=0)}  # return mean and variance as distribution parameters


    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor, horizon: int) -> torch.Tensor:
        """rollout a sequence of latent states given an initial state and a sequence of actions"""
        z_t = z_0
        predicted_states = []
        for t in range(horizon):
            action_t = actions[:, t, :]
            z_t = self.predict(z_t, action_t)
            predicted_states.append(z_t)
        return torch.stack(predicted_states, dim=1)  # Shape: (batch_size, horizon, latent_dim)
    
    def forward(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """alias for predict() to allow calling the dynamics model directly"""
        return self.predict(z_t, action_emb)