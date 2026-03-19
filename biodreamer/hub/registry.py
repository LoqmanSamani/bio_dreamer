"""
biodreamer.hub.registry — Unified Model Registry.

Purpose:
    Central registry that discovers, indexes, and loads models from multiple
    sources: local checkpoints, Hugging Face Hub (community models), and
    Hugging Face Hub (custom BioDreamer models). The web frontend's "Model Hub"
    page queries this registry.

Components to implement:
    - ModelRegistry:
        - list_available(source='all') → list of ModelInfo (name, source, size, description)
        - load_model(model_id, device) → instantiated PyTorch nn.Module
        - is_cached(model_id) → bool (is the model already downloaded locally?)
        - get_model_info(model_id) → ModelInfo (metadata, model card, metrics)

    - ModelInfo (dataclass):
        - model_id: str (e.g., "biodreamer/protein-dreamer-v1" or "facebook/esm2_t33_650M")
        - source: 'local' | 'huggingface' | 'custom'
        - model_type: 'encoder' | 'dynamics' | 'decoder' | 'reward' | 'policy' | 'world_model'
        - domain: 'mol_dreamer' | 'protein_dreamer' | 'cell_dreamer'
        - description: str
        - metrics: dict (evaluation scores)

    Pre-registered community models:
        - facebook/esm2_t33_650M_UR50D  (protein sequence encoder)
        - facebook/esm2_t36_3B_UR50D    (larger protein encoder)
        - facebook/esmfold_v1           (structure prediction oracle)
        - cborg/geneformer              (single-cell foundation model)

Design notes:
    - Uses huggingface_hub library for HF operations.
    - Local models are stored in MODEL_CACHE_DIR (from .env).
    - Model loading is lazy — models are instantiated only when requested.
    - Thread-safe for concurrent access from the FastAPI server.
"""
