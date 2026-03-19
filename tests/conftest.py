"""
conftest.py — Shared Pytest Fixtures for BioDreamer Test Suite.

Purpose:
    Provides reusable fixtures shared across all unit, integration, and
    e2e tests. Centralises test configuration so individual test files
    remain focused on assertions.

Fixtures to define:
    - device: torch.device — "cpu" for CI, "cuda" if available
    - sample_protein_sequence: str — short test sequence (e.g. 50 AA)
    - sample_mutations: list[dict] — example mutation list
    - sample_config: dict — minimal YAML config loaded from configs/
    - mock_world_model: MagicMock — mocked WorldModel with .imagine()
    - mock_encoder / mock_dynamics / mock_decoder: MagicMock components
    - tmp_checkpoint_dir: Path — temporary dir for saving/loading checkpoints
    - fastapi_test_client: TestClient — httpx-based async test client for server/
    - sample_pdb_data: str — minimal PDB file content for structure tests
    - sample_h5ad_path: Path — path to tiny test h5ad fixture file

Configuration:
    - pytest markers: @pytest.mark.slow, @pytest.mark.gpu, @pytest.mark.integration
    - Automatically skip GPU tests when CUDA unavailable
"""
