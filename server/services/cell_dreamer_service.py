"""
server.services.cell_dreamer_service — CellDreamer Business Logic.

Purpose:
    Orchestrates CellDreamer inference: loads world model, runs perturbation
    planning, formats results (UMAP coordinates, gene heatmaps).

Components to implement:
    - CellDreamerService:
        - submit_perturbation_job(request) → job_id
        - predict_perturbation(expression, perturbation) → predicted outcome
        - get_reference_cell_types() → list of available targets
"""
