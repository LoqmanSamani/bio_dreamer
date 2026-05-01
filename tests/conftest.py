"""Shared pytest fixtures for the BioDreamer test suite."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from biodreamer.protein_dreamer.data.dataset import AssayType
from biodreamer.protein_dreamer.tokenizers import MutationRecord


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring CUDA")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Protein sequences and mutations
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_protein_sequence() -> str:
    """50-residue canonical sequence using all 20 amino acids."""
    return "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL"


@pytest.fixture
def sample_mutations() -> list[MutationRecord]:
    return [
        MutationRecord(wt_aa="A", position=1, mut_aa="C", raw="A1C"),
        MutationRecord(wt_aa="G", position=6, mut_aa="K", raw="G6K"),
    ]


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

_WT = "ACDEFGHIKLMNPQRSTVWY"  # 20-residue wild-type for fixture DFs


@pytest.fixture
def sample_dms_dataframe() -> pd.DataFrame:
    """Minimal ProteinGym-style DataFrame with three rows of different assay types."""
    return pd.DataFrame([
        {
            "wt_sequence":    _WT,
            "mutation":       "A1C",
            "mutant_sequence": "C" + _WT[1:],
            "score":          0.5,
            "assay_id":       "STAB_001",
        },
        {
            "wt_sequence":    _WT,
            "mutation":       "C2D",
            "mutant_sequence": _WT[0] + "D" + _WT[2:],
            "score":          0.8,
            "assay_id":       "BIND_001",
        },
        {
            "wt_sequence":    _WT,
            "mutation":       "D3E",
            "mutant_sequence": _WT[:2] + "E" + _WT[3:],
            "score":          0.3,
            "assay_id":       "ACTV_001",
        },
    ])


@pytest.fixture
def sample_assay_type_map() -> dict[str, AssayType]:
    return {
        "STAB_001": AssayType.STABILITY,
        "BIND_001": AssayType.BINDING_AFFINITY,
        "ACTV_001": AssayType.CATALYTIC_ACTIVITY,
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Minimal PDB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdb_data() -> str:
    """Three-residue (ALA) PDB with backbone atoms only."""
    lines = []
    for i, (res, x) in enumerate([(1, 0.0), (2, 3.8), (3, 7.6)], start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{res:4d}    "
            f"{x:8.3f}   0.000   0.000  1.00  0.00           C"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock model components
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    enc = MagicMock()
    enc.get_latent_dim.return_value = 64
    enc.encode.return_value = torch.zeros(1, 64)
    return enc


@pytest.fixture
def mock_dynamics():
    dyn = MagicMock()
    dyn.predict.return_value = torch.zeros(1, 64)
    return dyn


@pytest.fixture
def mock_decoder():
    dec = MagicMock()
    dec.decode.return_value = {"sequence": torch.zeros(1, 20, 20)}
    return dec


@pytest.fixture
def mock_world_model(mock_encoder, mock_dynamics, mock_decoder):
    wm = MagicMock()
    wm.encoder = mock_encoder
    wm.dynamics = mock_dynamics
    wm.decoder = mock_decoder
    wm.encode.return_value = torch.zeros(1, 64)
    wm.imagine.return_value = [(torch.zeros(1, 64), torch.tensor(0.0))]
    return wm
