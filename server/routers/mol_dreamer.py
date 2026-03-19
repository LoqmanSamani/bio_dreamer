"""
server.routers.mol_dreamer — MolDreamer API Endpoints.

Purpose:
    REST API endpoints for the MolDreamer module. Handles molecular dynamics
    world model inference and molecular optimisation jobs.

Endpoints to implement:
    POST /api/mol-dreamer/submit
        Request: PDB file or molecular structure, target property, optimisation params
        Response: job_id

    GET /api/mol-dreamer/results/{job_id}
        Response: latent trajectory, property predictions, optimised coordinates

    POST /api/mol-dreamer/simulate
        Request: molecular structure, simulation parameters (time, temperature)
        Response: predicted trajectory in the world model (fast, no real MD)

Design notes:
    - Molecular structure input can be PDB, SDF, or MOL2 format.
    - Results include coordinate trajectories for the 3D molecular viewer
      (TrajectoryPlayer component in the frontend).
    - Property evolution plots (ΔG over trajectory steps) are returned as
      JSON arrays for the frontend PlotlyChart.
"""
