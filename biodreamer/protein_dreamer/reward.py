import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from biodreamer.core.reward import BaseRewardHead





class ProteinRewardHead(BaseRewardHead):
    """multi-objective reward head for protein fitness prediction.
    predicts multiple fitness dimensions (stability, affinity, activity,
    expressibility) from a shared latent representation of the protein.
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dim: int = 128,
        weights: Optional[Dict[str, float]] = None, 
        shared_head: Optional[Any] = None, 
        stab_head: Optional[Any] = None,
        affin_head: Optional[Any] = None,
        act_head: Optional[Any] = None,
        express_head: Optional[Any] = None,
        device: Optional[torch.device] = None 
        ) -> None:
        super().__init__(latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.weights = weights if weights is not None else {"stability": 0.4, "affinity": 0.3, "activity": 0.2, "expressibility": 0.1}
        self.device = device if device is not None and isinstance(device, torch.device) else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # shared layers
        self.shared_fc = shared_head.to(self.device) if shared_head is not None else nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        ).to(self.device)
        # stability head
        self.stab_head = stab_head.to(self.device) if stab_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # affinity head
        self.affin_head = affin_head.to(self.device) if affin_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # activity head
        self.act_head = act_head.to(self.device) if act_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # expressibility head
        self.express_head = express_head.to(self.device) if express_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)

    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        """predict a scalar reward by aggregating all fitness dimensions"""
        rewards = self.predict_multi(z_t)
        # weighted sum (weights can be tuned in config)
        scalar_reward = sum(self.weights[key] * rewards[key] for key in rewards)
        return scalar_reward.squeeze(-1)

    def predict_multi(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """predict individual fitness dimensions"""
        shared_rep = self.shared_fc(z_t)
        stab = self.stab_head(shared_rep).squeeze(-1)
        affin = self.affin_head(shared_rep).squeeze(-1)
        act = self.act_head(shared_rep).squeeze(-1)
        express = self.express_head(shared_rep).squeeze(-1)
        
        return {
            "stability": stab,
            "affinity": affin,
            "activity": act,
            "expressibility": express
        }