from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from biodreamer.core.decoder import BaseDecoder
from biodreamer.core.dynamics import BaseDynamics
from biodreamer.core.encoder import BaseEncoder
from biodreamer.core.reward import BaseRewardHead
from biodreamer.core.world_model import WorldModel

logger = logging.getLogger(__name__)


class ProteinWorldModel(WorldModel):
    """ProteinDreamer world model: encoder + dynamics + reward + optional decoder.

    Inherits encode(), imagine(), decode(), and predict_reward() from
    core.WorldModel. Adds compute_loss() to coordinate the JEPA dynamics
    objective (L_dyn) with the multi-head reward loss (L_rew).

    The decoder is optional — JEPA operates entirely in latent space.
    Pass one only when decoded outputs are needed (interpretability /
    validation / frontend serving).
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        dynamics: BaseDynamics,
        reward_head: BaseRewardHead,
        decoder: Optional[BaseDecoder] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            dynamics=dynamics,
            reward_head=reward_head,
            decoder=decoder,
        )

    def compute_loss(
        self,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
        z_t1_target: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        beta_dyn: float = 1.0,
        beta_rew: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Combined dynamics + reward training loss.

        Args:
            z_t:          Current latent state (B, latent_dim).
            action_emb:   Action embedding (B, action_dim).
            z_t1_target:  Target next latent from the EMA/target encoder (B, latent_dim).
            targets:      Fitness target dict — keys "stability", "affinity",
                          "activity", "fitness", each (B,); NaN where unmeasured.
            beta_dyn:     Weight on the dynamics loss (L_B / L_A).
            beta_rew:     Weight on the reward loss (L_rew).

        Returns:
            Dict with keys "loss", "loss_dyn", "loss_rew".
        """
        z_t1_pred = self.dynamics.predict(z_t, action_emb)
        loss_dyn = F.mse_loss(z_t1_pred, z_t1_target.detach())
        loss_rew = self.reward_head.compute_loss(z_t, targets)
        loss = beta_dyn * loss_dyn + beta_rew * loss_rew
        return {"loss": loss, "loss_dyn": loss_dyn, "loss_rew": loss_rew}
