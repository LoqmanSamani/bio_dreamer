"""
server.routers.models — Model Management API Endpoints.

Purpose:
    REST API for browsing, loading, and managing ML models. This powers the
    "Model Hub" page in the frontend where users can see available models
    from Hugging Face Hub and locally trained checkpoints.

Endpoints to implement:
    GET /api/models/
        Response: list of available models (local + HF Hub) with metadata

    GET /api/models/{model_id}
        Response: detailed model info (architecture, metrics, model card)

    POST /api/models/load
        Request: model_id, device
        Response: confirmation that model is loaded into memory

    POST /api/models/unload
        Request: model_id
        Response: confirmation that model is freed from GPU memory

    POST /api/models/upload
        Request: model_id, destination (HF Hub repo)
        Response: job_id (upload runs async)

    POST /api/models/download
        Request: HF Hub model_id
        Response: job_id (download + cache runs async)

Design notes:
    - Uses biodreamer.hub.registry.ModelRegistry as the backend.
    - Model loading is GPU-memory-aware: only load N models simultaneously
      (configured in server.yaml).
    - Community models (ESM-2, etc.) are listed from a hardcoded catalogue
      plus auto-discovery via HF Hub API search.
"""
