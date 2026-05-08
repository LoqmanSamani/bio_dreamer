from typing import Dict, Optional
import torch
import torch.nn as nn
from biodreamer.core.policy import BasePolicy




class ExpectedFreeEnergy:
    """Computes Expected Free Energy (EFE) for action selection.

    G(π) = E[ -r_t ] + E[ -η · Unc(z_t, a_t) ]

    The policy *minimises* G, so:
        - Higher predicted reward  → lower G  → preferred  (exploitation)
        - Higher model uncertainty → lower G  → preferred  (exploration)
    """

    @staticmethod
    def compute_efe(rewards: torch.Tensor, uncertainties: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """Compute per-action EFE scores.

        Args:
            rewards: Predicted rewards of shape (B, num_actions).
            uncertainties: Predictive uncertainties of shape (B, num_actions).
            eta: Weight for the epistemic (exploration) term.

        Returns:
            EFE scores of shape (B, num_actions).  Lower is better.
        """
        pragmatic = -rewards
        epistemic = -eta * uncertainties
        return pragmatic + epistemic

    @staticmethod
    def pragmatic_value(rewards: torch.Tensor) -> torch.Tensor:
        """Per-action pragmatic value (negative reward).

        Args:
            rewards: Predicted rewards of shape (B, num_actions).

        Returns:
            Pragmatic component of shape (B, num_actions).
        """
        return -rewards

    @staticmethod
    def epistemic_value(uncertainties: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        """Per-action epistemic value (negative scaled uncertainty).

        Args:
            uncertainties: Predictive uncertainties of shape (B, num_actions).
            eta: Exploration weight.

        Returns:
            Epistemic component of shape (B, num_actions).
        """
        return -eta * uncertainties




class ActiveInferencePolicy(BasePolicy):
    """Policy that selects actions by minimising Expected Free Energy.

    This is a concrete-enough base for all domain-specific Active Inference
    policies.  Subclasses must implement ``_generate_candidate_actions``
    to supply domain-appropriate action candidates (discrete mutations,
    continuous latent perturbations, etc.).

    The world model is stored as a reference (not a sub-module) so its
    parameters are not double-counted during optimisation.
    """
    def __init__(self, latent_dim: int, action_dim: int, world_model: nn.Module, eta: float = 1.0, n_samples: int = 20) -> None:
        """
        Args:
            latent_dim: Dimensionality of the latent state.
            action_dim: Dimensionality of action embeddings.
            world_model: A trained WorldModel instance (encoder + dynamics + reward_head).  
                Stored as a plain attribute so it is not registered as a child module.
            eta: Weight for the epistemic exploration term in EFE.
            n_samples: Number of stochastic dynamics samples used to
                estimate uncertainty (for diffusion-based dynamics).
        """
        super().__init__(latent_dim, action_dim)
        self.world_model = world_model
        self.eta = eta
        self.n_samples = n_samples

    def _generate_candidate_actions(self, z_t: torch.Tensor) -> torch.Tensor:
        """Generate candidate action embeddings for the current state.

        Subclasses should override this to produce meaningful candidates
        (e.g. all single-site mutations, sampled continuous perturbations).

        Args:
            z_t: Current latent state of shape (B, latent_dim).

        Returns:
            Candidate action embeddings of shape (B, num_actions, action_dim).
        """
        raise NotImplementedError(
            "_generate_candidate_actions() must be implemented by domain-specific subclasses."
        )

    def evaluate_actions(self, z_t: torch.Tensor, candidate_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score candidate actions by Expected Free Energy.

        For each candidate action the method:
            1. Predicts the next latent state z_{t+1} via the dynamics model.
            2. Predicts the reward from z_{t+1} via the reward head.
            3. Estimates predictive uncertainty from the dynamics model.
            4. Computes EFE = -reward - η·uncertainty.

        Args:
            z_t: Current latent state of shape (B, latent_dim).
            candidate_actions: Action embeddings of shape (B, A, action_dim)
                where A is the number of candidates.

        Returns:
            Dict with keys:
                'efe':            (B, A)  EFE scores (lower → better)
                'rewards':        (B, A)  predicted rewards from z_{t+1}
                'uncertainties':  (B, A)  dynamics uncertainty estimates
        """
        B, A, D_a = candidate_actions.shape
        z_expanded = z_t.unsqueeze(1).expand(-1, A, -1)

        # flatten for batched dynamics / reward calls
        z_flat = z_expanded.reshape(B * A, -1)
        a_flat = candidate_actions.reshape(B * A, D_a)
        
        z_next_flat = self.world_model.dynamics.predict(z_flat, a_flat)
        rewards_flat = self.world_model.reward_head.predict(z_next_flat)
        rewards = rewards_flat.view(B, A)
        
        try:
            # (n_samples, B*A, latent_dim) → variance → mean over latent dim
            samples = self.world_model.dynamics.predict_distribution(
                z_flat, a_flat, n_samples=self.n_samples
            )
            uncertainties_flat = samples.var(dim=0).mean(dim=-1)
        except NotImplementedError:
            # deterministic dynamics — uncertainty is zero
            uncertainties_flat = torch.zeros(B * A, device=z_t.device)

        uncertainties = uncertainties_flat.view(B, A)

        efe = ExpectedFreeEnergy.compute_efe(rewards, uncertainties, self.eta)

        return {
            "efe": efe,
            "rewards": rewards,
            "uncertainties": uncertainties,
        }

    def select_action(self, z_t: torch.Tensor) -> torch.Tensor:
        """Select the action minimising Expected Free Energy.

        Generates candidates via ``_generate_candidate_actions``, scores
        them, and returns the best one per batch element.

        Args:
            z_t: Current latent state of shape (B, latent_dim).

        Returns:
            Best action embedding of shape (B, action_dim).
        """
        candidate_actions = self._generate_candidate_actions(z_t)  # (B, A, action_dim)
        result = self.evaluate_actions(z_t, candidate_actions)
        best_idx = result["efe"].argmin(dim=-1)  # (B,)

        # gather the best action per batch element
        best_idx_expanded = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._action_dim)
        best_actions = candidate_actions.gather(1, best_idx_expanded).squeeze(1)
        return best_actions

    def select_action_with_exploration(self, z_t: torch.Tensor) -> torch.Tensor:
        """Sample an action proportional to softmin of EFE (Boltzmann).

        Instead of always picking the argmin, sample from a softmax
        distribution over negative EFE scores for training diversity.

        Args:
            z_t: Current latent state of shape (B, latent_dim).

        Returns:
            Sampled action embedding of shape (B, action_dim).
        """
        candidate_actions = self._generate_candidate_actions(z_t)
        result = self.evaluate_actions(z_t, candidate_actions)

        # Boltzmann selection: sample ∝ exp(-EFE)
        logits = -result["efe"]
        idx = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._action_dim)
        return candidate_actions.gather(1, idx_expanded).squeeze(1)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy on a batch of imagined trajectories.

        This base implementation is a no-op.  Domain-specific subclasses
        (e.g. ProteinActiveInferencePolicy) should override to implement
        actor-critic updates on imagined rollouts.

        Args:
            batch: Dict with at least 'states' (B, H, d) and
                'rewards' (B, H).

        Returns:
            Empty dict (override in subclass for real losses).
        """
        return {}
