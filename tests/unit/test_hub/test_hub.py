"""
test_hub.py — Unit Tests for biodreamer.hub (registry, upload, download, model_card).

Tests to implement:
    - test_registry_list_models: lists pre-registered community models
    - test_registry_load_model: loads a model by ID (mocked HF download)
    - test_upload_to_hub: serialises model + config + card, calls HF API (mocked)
    - test_download_from_hub: pulls model, caches locally (mocked)
    - test_model_card_generation: produces valid markdown from config + metrics
    - test_version_pinning: specific revision is used when requested
    - test_cache_hit: second download uses cached checkpoint
"""
