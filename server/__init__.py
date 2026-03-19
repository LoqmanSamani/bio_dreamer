"""
server — BioDreamer FastAPI Backend.

This package implements the REST API that connects the frontend web application
to the BioDreamer ML core. It handles:
    - Receiving design requests (protein sequence, PDB, expression data)
    - Dispatching inference/training jobs to GPU workers
    - Streaming results back to the frontend
    - Managing models (list, load, upload to HF Hub)
    - Job lifecycle management (submit, track, cancel)
"""
