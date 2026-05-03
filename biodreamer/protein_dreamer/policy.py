from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from biodreamer.core.active_inference import ActiveInferencePolicy

logger = logging.getLogger(__name__)





class ProteinActiveInferencePolicy(ActiveInferencePolicy):
    """active inference policy for single-site mutation planning.

    generates all L×n_aa candidate substitutions at each step via an
    ActionEncoder, scores them by Expected Free Energy (EFE), and selects
    the mutation that minimises EFE (exploitation + exploration).

    set_sequence() must be called before each trajectory to register
    the current wt amino acid state used to build candidate mutations.
    """
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        world_model: nn.Module,
        action_encoder: nn.Module,
        seq_len: int,
        n_aa: int = 20,
        eta: float = 1.0,
        n_samples: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            world_model=world_model,
            eta=eta,
            n_samples=n_samples,
        )
        self.device = (
            device if device is not None and isinstance(device, torch.device)
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.action_encoder = action_encoder
        self.seq_len = seq_len
        self.n_aa = n_aa
        # fixed candidate index grids, built once, moved with the module
        positions = torch.arange(seq_len, dtype=torch.long).repeat_interleave(n_aa)
        aa_new    = torch.arange(n_aa,    dtype=torch.long).repeat(seq_len)
        self.register_buffer("_positions", positions)   # (L*n_aa,)
        self.register_buffer("_aa_new",    aa_new)      # (L*n_aa,)
        # wt amino acid per position, updated via set_sequence()
        self.register_buffer("_wt_aa", torch.zeros(seq_len, dtype=torch.long))

    def set_sequence(self, aa_indices: torch.Tensor) -> None:
        """update the wt amino acid indices used to generate mutation candidates.
        args:
            aa_indices: LongTensor of shape (L,) with 0-based aa index per residue.
        """
        self._wt_aa = aa_indices.to(self.device)

    def _generate_candidate_actions(self, z_t: torch.Tensor) -> torch.Tensor:
        """enumerate all L×n_aa single-site substitutions as action embeddings"""
        B = z_t.shape[0]
        n_cands = self.seq_len * self.n_aa
        aa_old = self._wt_aa.repeat_interleave(self.n_aa)  # (L*n_aa,)
        action = {
            "position": self._positions.unsqueeze(0).expand(B, -1).reshape(-1),
            "aa_old":   aa_old.unsqueeze(0).expand(B, -1).reshape(-1),
            "aa_new":   self._aa_new.unsqueeze(0).expand(B, -1).reshape(-1),
        }
        embs = self.action_encoder(action)              # (B*n_cands, action_dim)
        return embs.view(B, n_cands, self._action_dim)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """reinforce policy gradient on imagined rollout rewards.

        args:
            batch: must contain:
                'log_probs': (B, H) log-probability of the selected action at each step.
                'rewards':   (B, H) predicted reward at each step.
        """
        log_probs = batch["log_probs"]  # (B, H)
        rewards   = batch["rewards"]    # (B, H)
        returns   = rewards.flip(1).cumsum(1).flip(1)
        loss = -(log_probs * returns.detach()).mean()
        return {"policy_loss": loss}
