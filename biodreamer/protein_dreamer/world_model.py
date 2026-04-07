from typing import Any, Dict    

import torch    
import torch.nn as nn





class WorldModel(nn.Module):
    """Composition class that wires encoder + dynamics + decoder + reward into a unified interface."""
    def __init__(
        self,
        encoder: nn.Module,
        dynamics: nn.Module,
        decoder: nn.Module,
        reward_head: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.reward_head = reward_head
    
    def encode(self, observation: Any) -> torch.Tensor:
        """Encode an observation into a latent state z_t."""
        return self.encoder(observation)
    
    def imagine(self, z_t: torch.Tensor, actions: Any) -> Dict[str, torch.Tensor]:
        """Imagine a trajectory of future latent states and rewards given an initial state and action sequence."""
        # This is a placeholder implementation. The actual logic will depend on the specific dynamics and reward head.
        trajectory = self.dynamics(z_t, actions)
        rewards = self.reward_head(trajectory['states'])
        return {'states': trajectory['states'], 'rewards': rewards}
    
    def decode(self, z_t: torch.Tensor) -> Any:
        """Decode a latent state back into an observation."""
        return self.decoder(z_t)
    
    def predict_reward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Predict the reward for a given latent state."""
        return self.reward_head(z_t)
    
    def forward(self, observation: Any, actions: Any) -> Dict[str, torch.Tensor]:
        """Full forward pass through the world model: encode → imagine → decode → predict reward."""
        z_t = self.encode(observation)
        trajectory = self.imagine(z_t, actions)
        decoded_observations = [self.decode(state) for state in trajectory['states']]
        predicted_rewards = [self.predict_reward(state) for state in trajectory['states']]
        return {
            'decoded_observations': decoded_observations,
            'predicted_rewards': predicted_rewards
        }