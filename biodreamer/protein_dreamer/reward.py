from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from biodreamer.core.reward import BaseRewardHead
from biodreamer.protein_dreamer.config import ProteinDreamerConfig

logger = logging.getLogger(__name__)


class ProteinRewardHead(BaseRewardHead):
    """multi-objective reward head predicting four fitness dimensions from a shared latent.

    heads:
      stability: ΔΔG / Tm (thermodynamic stability)
      affinity: Kd (binding affinity)
      activity: kcat (catalytic activity)
      fitness: organismal/holistic fitness (growth, survival, DMS enrichment scores)

    the fitness head should use a lower weight during backbone pretraining and a higher weight
    during per-protein fine-tuning, because organismal fitness and ΔΔG can be
    anticorrelated, routing organismal fitness to the stability head creates gradient
    conflict.
    """
    def __init__(
        self,
        config: Any = None,
        shared_backbone: Optional[Any] = None,
        stab_head: Optional[Any] = None,
        affin_head: Optional[Any] = None,
        act_head: Optional[Any] = None,
        fitness_head: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if config is None:
            config = ProteinDreamerConfig().default()["reward"]
        latent_dim = config.get("latent_dim", 256)
        hidden_dim = config.get("hidden_dim", 512)
        weights_cfg = config.get("weights", {})
        weights = dict(weights_cfg) if weights_cfg else {
            "stability": 0.35,
            "affinity":  0.25,
            "activity":  0.20,
            "fitness":   0.20,
        }
        super().__init__(latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.weights = weights
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.shared_backbone = (
            shared_backbone.to(self.device) if shared_backbone is not None
            else nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
            ).to(self.device)
        )
        self.stab_head = (
            stab_head.to(self.device) if stab_head is not None
            else nn.Linear(self.hidden_dim // 2, 1).to(self.device)
        )
        self.affin_head = (
            affin_head.to(self.device) if affin_head is not None
            else nn.Linear(self.hidden_dim // 2, 1).to(self.device)
        )
        self.act_head = (
            act_head.to(self.device) if act_head is not None
            else nn.Linear(self.hidden_dim // 2, 1).to(self.device)
        )
        self.fitness_head = (
            fitness_head.to(self.device) if fitness_head is not None
            else nn.Linear(self.hidden_dim // 2, 1).to(self.device)
        )

    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        """scalar reward = weighted sum of all four fitness dimensions"""
        rewards = self.predict_multi(z_t)
        return sum(self.weights[k] * rewards[k] for k in rewards).squeeze(-1)

    def predict_multi(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """per-head predictions; keys match the targets dict from the data pipeline"""
        shared = self.shared_backbone(z_t.to(self.device))
        return {
            "stability": self.stab_head(shared).squeeze(-1),
            "affinity":  self.affin_head(shared).squeeze(-1),
            "activity":  self.act_head(shared).squeeze(-1),
            "fitness":   self.fitness_head(shared).squeeze(-1),
        }

    def compute_loss(
        self,
        z_t: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """masked SmoothL1 loss, NaN entries in targets are excluded per head"""
        preds = self.predict_multi(z_t)
        loss = torch.tensor(0.0, device=self.device)
        for k, w in self.weights.items():
            y = targets.get(k)
            if y is None:
                continue
            mask = ~torch.isnan(y)
            if mask.any():
                loss = loss + w * F.smooth_l1_loss(preds[k][mask], y[mask].to(self.device))
        return loss
