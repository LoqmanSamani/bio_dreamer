"""
server.services.mol_dreamer_service — MolDreamer Business Logic.

Purpose:
    Orchestrates MolDreamer inference: loads world model, runs latent MD
    simulation, formats trajectory results for the frontend.

Components to implement:
    - MolDreamerService:
        - submit_simulation_job(request) → job_id
        - predict_property(structure, property_name) → prediction
"""
