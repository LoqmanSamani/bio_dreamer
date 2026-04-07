"""
biodreamer.core.policy — Base RL policy interface.

Purpose:
    Abstract base class for RL policies that select actions (mutations,
    perturbations, parameter changes) to optimise rewards within the
    learned world model.

Components:
    - BasePolicy(nn.Module):
        - select_action(z_t) → action
        - select_action_with_exploration(z_t) → action (with entropy/noise)
        - get_action_distribution(z_t) → distribution over actions
        - update(trajectories) → loss dict (for on-policy methods)

    Concrete policy types to support:
        - LatentActorCritic:  SAC / PPO operating in continuous latent action space
        - DiscreteMutationPolicy: PPO / DQN for discrete mutation actions
                                  (which position × which amino acid)
        - MCTSPlanner: Monte Carlo Tree Search over mutation trees using
                       the world model for rollout evaluation
        - GFlowNetPolicy: Diversity-seeking alternative that samples
                          proportionally to reward

Design notes:
    - Policies are trained entirely "in imagination" — they act in the world
      model, not in the real environment. This is the key advantage of MBRL.
    - The MCTS planner uses the dynamics model for forward simulation and
      the reward head for leaf evaluation — no separate value network needed.
    - For ProteinDreamer, the action space is a structured discrete space:
      (position ∈ {1..L}, amino_acid ∈ {A..Y}, edit_type ∈ {sub, ins, del}).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BasePolicy(ABC, nn.Module):
    """Abstract base class for all RL policies in BioDreamer.

    Subclasses must implement:
        - select_action(z_t) → action tensor
        - update(batch) → dict of loss values
    """

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._action_dim = action_dim

    @abstractmethod
    def select_action(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select the best action given the current latent state.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            Action tensor (format depends on domain).
        """
        ...

    def select_action_with_exploration(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select an action with added exploration noise / entropy.

        Default implementation delegates to select_action().
        Subclasses should override to add stochasticity for training.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            Action tensor with exploration noise.
        """
        return self.select_action(z_t)

    def get_action_distribution(self, z_t: torch.Tensor) -> Any:
        """Return a distribution over actions for the given latent state.

        Optional — subclasses that maintain an explicit policy distribution
        (e.g. PPO, SAC) should override this.

        Args:
            z_t: Latent state tensor of shape (B, latent_dim).

        Returns:
            A torch.distributions.Distribution or equivalent.
        """
        raise NotImplementedError(
            "get_action_distribution() is optional; override in subclass."
        )

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy parameters given a batch of experience.

        Args:
            batch: Dict with keys like 'z_t', 'actions', 'rewards',
                'next_z_t', etc.  Exact keys depend on the algorithm.

        Returns:
            Dict of scalar loss / metric values for logging.
        """
        ...

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """nn.Module forward pass — delegates to select_action()."""
        return self.select_action(z_t)
