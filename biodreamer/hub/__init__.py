"""
biodreamer.hub — Hugging Face Hub Integration.

This sub-package manages model versioning, storage, and sharing via
Hugging Face Hub. It supports:
    - Downloading pre-trained community models (ESM-2, ProteinMPNN, scGPT, etc.)
    - Uploading custom-trained BioDreamer models (world models, policies)
    - A unified model registry that abstracts over local and remote models

Modules:
    - registry:    Unified model registry (discover, load, cache models)
    - upload:      Push trained models + model cards to Hugging Face Hub
    - download:    Pull models from HF Hub with caching and version pinning
    - model_card:  Auto-generate Hugging Face model card metadata
"""
