"""
server.services.protein_dreamer_service — ProteinDreamer Business Logic.

Purpose:
    Orchestrates ProteinDreamer inference: validates input, loads the correct
    world model + policy, runs the inference pipeline, and formats results.

Components to implement:
    - ProteinDreamerService:
        - submit_design_job(request) → job_id (dispatches to worker)
        - evaluate_single(sequence, model_id) → fitness prediction
        - get_available_landscapes() → list of ProteinGym assays
        - upload_experimental_data(data) → triggers active learning round

Design notes:
    - Separates API concerns (routers) from ML logic (this service).
    - Service instances are injected into routers via FastAPI dependency injection.
    - Holds reference to the ModelRegistry for model loading.
"""
