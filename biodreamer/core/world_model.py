from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from biodreamer.core.decoder import BaseDecoder
from biodreamer.core.dynamics import BaseDynamics
from biodreamer.core.encoder import BaseEncoder
from biodreamer.core.reward import BaseRewardHead




class WorldModel(nn.Module):
    """Composes encoder + dynamics + decoder + reward into a unified world model.

    The decoder is optional — it is NOT used during imagination rollouts
    (JEPA operates entirely in latent space). It is only needed for
    interpretability, validation, and serving decoded outputs to the frontend.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        dynamics: BaseDynamics,
        reward_head: BaseRewardHead,
        decoder: Optional[BaseDecoder] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward_head = reward_head
        self.decoder = decoder

    def encode(self, observation: Any) -> torch.Tensor:
        """Encode a raw observation into a latent state z_t.

        Args:
            observation: Domain-specific input forwarded to the encoder.

        Returns:
            Latent state tensor of shape (B, latent_dim).
        """
        return self.encoder.encode(observation)

    def imagine(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        reward_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Roll out a trajectory in imagination (latent space).

        Args:
            z_0: Initial latent state of shape (B, latent_dim).
            actions: Action embeddings of shape (B, H, action_dim) where
                H is the planning horizon.
            reward_threshold: If set, stop the rollout early once the
                predicted reward exceeds this value.

        Returns:
            Dict with keys:
                'states':  (B, H, latent_dim) predicted latent trajectory
                'rewards': (B, H) predicted rewards at each step
        """
        batch_size, horizon, _ = actions.shape
        states: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []

        z_t = z_0
        for t in range(horizon):
            action_t = actions[:, t, :]
            z_t = self.dynamics.predict(z_t, action_t)
            r_t = self.reward_head.predict(z_t)
            states.append(z_t)
            rewards.append(r_t)

            if reward_threshold is not None and r_t.mean().item() > reward_threshold:
                break

        return {
            "states": torch.stack(states, dim=1),
            "rewards": torch.stack(rewards, dim=1),
        }

    def decode(self, z_t: torch.Tensor) -> Any:
        """Decode a latent state into a reconstructed observation.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Reconstructed observation in domain-specific format.

        Raises:
            RuntimeError: If no decoder was provided at construction time.
        """
        if self.decoder is None:
            raise RuntimeError("WorldModel has no decoder — pass one at construction.")
        return self.decoder.decode(z_t)

    def predict_reward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict a scalar reward from a latent state.

        Args:
            z_t: Latent state tensor of shape (..., latent_dim).

        Returns:
            Scalar reward tensor.
        """
        return self.reward_head.predict(z_t)
