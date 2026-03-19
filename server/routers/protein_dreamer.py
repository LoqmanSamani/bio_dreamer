"""
server.routers.protein_dreamer — ProteinDreamer API Endpoints.

Purpose:
    REST API endpoints for the ProteinDreamer module. Handles protein design
    job submission, progress tracking, and result retrieval.

Endpoints to implement:
    POST /api/protein-dreamer/submit
        Request: wild-type sequence, target objective(s), mutation budget,
                 imagination horizon, model selection
        Response: job_id (async job dispatched to worker)

    GET /api/protein-dreamer/results/{job_id}
        Response: ranked mutation trajectories, fitness predictions,
                  structure visualisation data, uncertainty estimates

    POST /api/protein-dreamer/evaluate
        Request: specific mutant sequence
        Response: predicted fitness + structure (quick single-candidate evaluation)

    POST /api/protein-dreamer/active-learning/upload
        Request: experimental fitness measurements for proposed candidates
        Response: acknowledgement, triggers world model fine-tuning

    GET /api/protein-dreamer/landscapes
        Response: list of available pre-loaded fitness landscapes (ProteinGym)

Design notes:
    - Long-running jobs (dreaming, training) are dispatched to the worker queue
      and tracked via /api/jobs/{job_id}. Only lightweight operations run synchronously.
    - File uploads (PDB, FASTA) accepted via multipart form data.
    - Results include JSON data for the frontend visualisations (3D structure,
      sequence alignment, fitness plots).
"""
