"""
Config loading infrastructure for ProteinDreamer.

Usage:
    cfg = ProteinDreamerConfig.default()
    encoder = ProteinEncoder(cfg.encoder)
    dynamics = EnergyBasedDynamics(cfg.dynamics)

    constants = ProteinDreamerConstants.load()
    AA_LIST = list(constants["amino_acids"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_CONFIGS_DIR = Path(__file__).parents[2] / "configs" / "protein_dreamer"


class Namespace(dict):
    """Dict subclass with attribute-style access.

    cfg["key"] and cfg.key are equivalent.  Nested dicts are converted
    automatically by ProteinDreamerConfig.from_yaml().
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No config key '{key}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"Namespace({{{inner}}})"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts (and lists of dicts) to Namespace objects."""
    if isinstance(obj, dict):
        return Namespace({k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


class ProteinDreamerConfig:
    """Loader for ProteinDreamer YAML configuration files.

    Returns Namespace trees that support both dict-style (cfg["key"])
    and attribute-style (cfg.key) access.
    """

    _DEFAULT_PATH: Path = _CONFIGS_DIR / "default.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Namespace:
        """Load a YAML file and return a Namespace tree."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return _to_namespace(raw or {})

    @classmethod
    def default(cls) -> Namespace:
        """Return the bundled default configuration."""
        return cls.from_yaml(cls._DEFAULT_PATH)


class ProteinDreamerConstants:
    """Loader for protein domain constants (amino acid alphabet, etc.).

    Constants are cached after first load so repeated calls are free.
    """

    _PATH: Path = _CONFIGS_DIR / "constants.yaml"
    _cache: Optional[Namespace] = None

    @classmethod
    def load(cls) -> Namespace:
        """Return the constants Namespace (cached after first call)."""
        if cls._cache is None:
            with open(cls._PATH) as fh:
                cls._cache = _to_namespace(yaml.safe_load(fh) or {})
        return cls._cache

    @classmethod
    def amino_acids(cls) -> List[str]:
        """Return the ordered list of 20 canonical amino acids."""
        return list(cls.load()["amino_acids"])

    @classmethod
    def aa_to_idx(cls) -> Dict[str, int]:
        """Return {aa: 0-based index} mapping."""
        return {aa: i for i, aa in enumerate(cls.amino_acids())}

    @classmethod
    def n_aa(cls) -> int:
        """Return the number of canonical amino acids (20)."""
        return int(cls.load()["n_aa"])
