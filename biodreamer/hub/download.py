"""
biodreamer.hub.download — Pull Models from Hugging Face Hub.

Purpose:
    Downloads pre-trained models from Hugging Face Hub with local caching,
    version pinning, and automatic integrity verification.

Components to implement:
    - download_model(model_id, revision='main', cache_dir=None) → local path
    - load_from_hub(model_id, model_class, device='cpu') → instantiated nn.Module
    - ensure_cached(model_id) → bool (download if not cached)
    - get_latest_revision(model_id) → str (commit hash of latest version)
    - clear_cache(model_id=None) → free disk space (specific model or all)

    Supported community models:
        - ESM-2 family (via transformers library)
        - ESMFold (via esm library or transformers)
        - ProteinMPNN (custom loading from published weights)
        - scGPT / Geneformer (via transformers or custom)

Design notes:
    - Cache directory: MODEL_CACHE_DIR from .env (default: ./model_cache).
    - Version pinning: lock to specific git revisions for reproducibility.
    - Automatic retry with exponential backoff for flaky network connections.
    - Integration with the ModelRegistry — downloaded models are automatically
      registered in the local model index.
"""
