from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np





class BaseOracle(ABC):
    """Abstract base class for fitness / objective oracles in BioDreamer.

    An oracle maps a biological entity (protein sequence, SMILES string, cell
    state) to a scalar fitness value.  Returns float('nan') when the fitness is
    unknown or cannot be computed.

    All three sub-projects inherit from this:
      - ProteinDreamer: sequence → ΔΔG, Kd, kcat, enrichment score
      - MolDreamer:     SMILES / conformation → docking score, QED
      - CellDreamer:    gene expression vector → phenotypic fitness

    Subclasses must implement:
        query(entity) → float
    """

    @abstractmethod
    def query(self, entity: Any) -> float:
        """Return a scalar fitness score for the given biological entity."""

    def query_multi(self, entities: List[Any]) -> Dict[Any, float]:
        """Batch query; default falls back to per-entity query()."""
        return {e: self.query(e) for e in entities}





class BaseEnvironment(ABC):
    """Abstract base class for BioDreamer environments.

    Wraps a biological fitness landscape as a gym-like MDP for RL training
    and evaluation.  Provides ground-truth transitions that the world model
    learns to approximate.

    The environment is the ground truth and is used only for:
      - World model training (ground-truth transitions from real data)
      - Final policy evaluation
      - Active-learning data collection

    During policy training the agent operates entirely in the world model
    (dreaming) — the environment is NOT called.

    Subclasses must implement:
        reset() → initial state
        step(action) → (next_state, reward, done, info)
        render() → visualisation dict
    """

    @abstractmethod
    def reset(self) -> Any:
        """Reset the episode and return the initial state."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply an action and return (next_state, reward, done, info)."""

    @abstractmethod
    def render(self) -> Dict[str, Any]:
        """Return a plain dict of state information for logging / visualisation."""
        
        
        


class BaseStructureOracle(ABC):
    """abstract base for protein structure oracles.

    returns (coords, plddt, ptm) compatible with ProteinEncoder.embed_observation().
    """

    @abstractmethod
    def predict(
        self, sequence: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
        """predict protein structure.

        returns:
            coords: (L, 3) Cα coordinates in Å
            plddt:  (L,) per-residue confidence 0–100, or None
            ptm:    global pTM score 0–1, or None
        """
