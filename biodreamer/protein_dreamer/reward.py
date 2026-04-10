"""
biodreamer.protein_dreamer.reward — Multi-Objective Protein Fitness Predictor.

Purpose:
    Predicts fitness scores from protein latent states — the reward signal
    that the RL policy optimises. Supports multi-objective optimisation
    (stability + function + expressibility simultaneously).

Components to implement:
    - ProteinRewardHead(BaseRewardHead):
        Multi-task prediction heads branching from the shared latent state:
        - predict_stability(z_t) → ΔΔG (kcal/mol) — thermodynamic stability change
        - predict_affinity(z_t) → Kd or IC50 — binding affinity
        - predict_activity(z_t) → kcat or relative activity — catalytic function
        - predict_expressibility(z_t) → solubility / expression level score
        - predict_multi(z_t) → dict of all fitness dimensions
        - scalarise(rewards_dict, weights) → single scalar reward

    Training data:
        - ProteinGym DMS assays (200+ proteins, diverse fitness measurements)
        - Tsuboyama mega-scale ΔΔG data (500k+ measurements)
        - BRENDA / EnzML enzyme activity data

    Multi-objective scalarisation strategies:
        - Weighted linear sum (configurable weights in YAML)
        - Tchebycheff scalarisation
        - Pareto-based (return full reward vector, let policy handle trade-offs)

Design notes:
    - Each head should output both a point prediction and an uncertainty estimate.
    - The reward head is fine-tuned per protein family for maximum accuracy.
    - Reward shaping: optionally add a novelty bonus (distance from wild-type in
      latent space) to encourage diverse exploration.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from biodreamer.core.reward import BaseRewardHead




class ProteinRewardHead(BaseRewardHead):
    """Multi-objective reward head for protein fitness prediction.

    Predicts multiple fitness dimensions (stability, affinity, activity,
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
        self.device = device if device is not None else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Shared layers
        self.shared_fc = shared_head.to(self.device) if shared_head is not None else nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        ).to(self.device)
        # Stability head
        self.stability_head = stab_head.to(self.device) if stab_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # Affinity head
        self.affinity_head = affin_head.to(self.device) if affin_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # Activity head
        self.activity_head = act_head.to(self.device) if act_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)
        # Expressibility head
        self.expressibility_head = express_head.to(self.device) if express_head is not None else nn.Linear(
            self.hidden_dim // 2, 1).to(self.device)

    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict a scalar reward by aggregating all fitness dimensions."""
        rewards = self.predict_multi(z_t)
        # weighted sum (weights can be tuned in config)
        scalar_reward = sum(self.weights[key] * rewards[key] for key in rewards)
        return scalar_reward.squeeze(-1)

    def predict_multi(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict individual fitness dimensions."""
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
