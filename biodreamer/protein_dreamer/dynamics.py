"""
biodreamer.protein_dreamer.dynamics — Protein Fitness Landscape Dynamics Model.

Purpose:
    The core "world model" for ProteinDreamer. Predicts how the protein's
    latent state (encoding sequence, structure, and fitness) changes after
    a mutation is applied:
        z_{t+1} = f_θ(z_t, mutation_t)

    This allows the agent to "dream" — simulate entire mutation trajectories
    without querying expensive structure predictors or wet-lab assays at each step.

Components to implement:
    - ProteinDynamics(BaseDynamics):
        Mutation-conditioned latent transition model. Options:

        1. Conditional Transformer:
           Input: (z_t, mutation_embedding) → transformer layers → z_{t+1}
           Mutation embedding encodes: (position, wild-type AA, mutant AA, edit_type)

        2. Latent Diffusion:
           Denoising diffusion in latent space conditioned on (z_t, mutation).
           Produces a distribution over z_{t+1}, capturing multi-modal outcomes
           (a mutation can have multiple structural consequences).

        3. RSSM (DreamerV3-style):
           Deterministic path: h_{t+1} = GRU(h_t, z_t, mutation)
           Stochastic state: z_{t+1} ~ q(z | h_{t+1})
           Combines deterministic tracking of mutation history with stochastic
           modelling of structural uncertainty.

    - MutationEmbedding:
        Encodes a mutation action into a fixed-size vector:
        (position_encoding, AA_embedding[wt], AA_embedding[mut], edit_type_embedding)

Design notes:
    - The dynamics model is trained on DMS data: input = (wild-type latent, mutation)
      → target = mutant fitness (and optionally mutant latent from ESMFold structure).
    - Multi-step rollouts accumulate mutations: z_0 → z_1 → ... → z_T.
      The model must handle compounding errors gracefully (symlog predictions,
      latent regularisation, truncated rollout training).
    - Epistasis modelling: the dynamics model implicitly captures epistatic effects
      because it conditions on the current state z_t (which encodes all prior mutations),
      not just the wild-type.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from biodreamer.core.dynamics import BaseDynamics



class ProteinDynamics(BaseDynamics):
    """Mutation-conditioned latent transition model for ProteinDreamer."""
    def __init__(self, trans_model: Any) -> None:
        super().__init__()
        self.trans_model = trans_model
       
        
    def predict(self, z_t: torch.Tensor, action_emb: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Predict the next latent state z_{t+1} given current state z_t and action embedding."""
        return self.predict_distribution(z_t, action_emb, n_samples)
    
    
    def predict_distribution(self, z_t: torch.Tensor, action_emb: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:    
        """Predict a distribution over next latent states z_{t+1} for uncertainty estimation."""
        input_emb = torch.cat([z_t, action_emb], dim=-1)
        z_nexts = [self.trans_model(input_emb) for _ in range(n_samples)]
        z_next_dist = torch.stack(z_nexts, dim=0)  # Shape: (n_samples, batch_size, latent_dim)
        return {"mean": torch.mean(z_next_dist, dim=0), "var": torch.var(z_next_dist, dim=0)}  # Return mean and variance as distribution parameters


    def rollout(self, z_0: torch.Tensor, actions: torch.Tensor, horizon: int) -> torch.Tensor:
        """Rollout a sequence of latent states given an initial state and a sequence of actions."""
        z_t = z_0
        predicted_states = []
        for t in range(horizon):
            action_t = actions[:, t, :]
            z_t = self.predict(z_t, action_t)
            predicted_states.append(z_t)
        return torch.stack(predicted_states, dim=1)  # Shape: (batch_size, horizon, latent_dim)
    
    def forward(self, z_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Alias for predict() to allow calling the dynamics model directly."""
        return self.predict(z_t, action_emb)
    
    



