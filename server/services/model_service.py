"""
server.services.model_service — Model Management Service.

Purpose:
    Backend for the Model Hub page. Manages model lifecycle: discovery,
    loading, unloading, uploading to Hugging Face Hub.

Components to implement:
    - ModelService:
        - list_models(filters) → list of ModelInfo
        - load_model(model_id, device) → loaded model reference
        - unload_model(model_id) → freed GPU memory
        - upload_model(model_id, hf_repo) → job_id (async upload)
        - download_model(hf_model_id) → job_id (async download)
        - get_gpu_memory_usage() → dict (used, available per device)

Design notes:
    - Wraps biodreamer.hub.registry.ModelRegistry.
    - Tracks GPU memory to prevent OOM errors when loading multiple models.
    - Thread-safe model loading/unloading for concurrent API requests.
"""
