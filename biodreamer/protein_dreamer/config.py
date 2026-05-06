"""
Config loading infrastructure for ProteinDreamer.

Usage:
    # Load the default config
    cfg = ProteinDreamerConfig.default()

    # Load a named experiment config (merges over default)
    cfg = ProteinDreamerConfig.load("small")

    # Pass sections directly to classes
    encoder  = ProteinEncoder(cfg["encoder"])
    dynamics = EnergyBasedDynamics(cfg["dynamics"])

    # Access domain constants
    constants = ProteinDreamerConstants.load()
    AA_LIST   = list(constants["amino_acids"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged rather than replaced; all other types are
    overwritten by the override value.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class ProteinDreamerConfig:
    """Loader for ProteinDreamer YAML configuration files.

    Returns Namespace trees that support both dict-style (cfg["key"])
    and attribute-style (cfg.key) access.

    Named configs live in configs/protein_dreamer/<name>.yaml.
    When loaded via load(), they are recursively merged on top of default.yaml
    so only the keys that differ need to be specified in the named file.
    """

    _DEFAULT_PATH: Path = _CONFIGS_DIR / "default.yaml"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Namespace:
        """Load a YAML file and return a Namespace tree (no merging)."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return _to_namespace(raw or {})

    @classmethod
    def default(cls) -> Namespace:
        """Return the bundled default configuration."""
        return cls.from_yaml(cls._DEFAULT_PATH)

    @classmethod
    def merge(cls, base: Namespace, override: Union[str, Path, Dict]) -> Namespace:
        """Return a new Namespace with *override* merged on top of *base*.

        *override* can be:
          - a dict / Namespace (merged directly)
          - a file path (loaded then merged)
          - a string without path separators (treated as a named config:
            configs/protein_dreamer/<name>.yaml)
        """
        if isinstance(override, str) and "/" not in override and "\\" not in override:
            override_path = _CONFIGS_DIR / f"{override}.yaml"
            with open(override_path) as fh:
                override_dict = yaml.safe_load(fh) or {}
        elif isinstance(override, (str, Path)):
            with open(override) as fh:
                override_dict = yaml.safe_load(fh) or {}
        else:
            override_dict = dict(override)

        merged = _deep_merge(dict(base), override_dict)
        return _to_namespace(merged)

    @classmethod
    def load(cls, name: str) -> Namespace:
        """Load a named config merged over the defaults.

        Equivalent to:  ProteinDreamerConfig.merge(default(), name)

        Example:
            cfg = ProteinDreamerConfig.load("small")
        """
        return cls.merge(cls.default(), name)


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
