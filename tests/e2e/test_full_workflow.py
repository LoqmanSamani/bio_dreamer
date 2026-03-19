"""
test_full_workflow.py — End-to-End Tests for Complete User Workflows.

Purpose:
    Simulates full user workflows from the API level: submit a job,
    poll until completion, retrieve results, and validate output format.
    These tests require a running backend (or use the TestClient with
    mocked GPU workers).

Tests to implement:
    - test_protein_design_workflow:
        1. POST /api/models/load (load a dummy world model)
        2. POST /api/protein-dreamer/submit (submit design task)
        3. Poll GET /api/jobs/{id} until status == "completed"
        4. GET /api/protein-dreamer/results/{id}
        5. Validate: candidates list is non-empty, fitness values are floats,
           mutations are valid format, structure PDB is parseable

    - test_mol_simulation_workflow:
        1. Load model → submit simulation → poll → get results
        2. Validate: trajectory frames count matches horizon, energies are finite

    - test_cell_perturbation_workflow:
        1. Load model → submit perturbation plan → poll → get results
        2. Validate: plans are ranked, gene effects have valid p-values

    - test_active_learning_workflow:
        1. Complete protein design → POST /api/protein-dreamer/active-learning
        2. Validate: proposed experiments have expected information gain > 0

    - test_model_upload_download_workflow:
        1. Train dummy model (mocked) → upload to HF Hub (mocked)
        2. Download from HF Hub (mocked) → load → verify predictions match
"""
