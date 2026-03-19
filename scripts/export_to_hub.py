"""
export_to_hub.py — Model Export to Hugging Face Hub.

Purpose:
    Packages a trained BioDreamer model (world model and/or policy) and
    uploads it to Hugging Face Hub with auto-generated model card,
    config, and safetensors weights.

Usage:
    python scripts/export_to_hub.py \\
        --checkpoint checkpoints/protein_dreamer/world_model.pt \\
        --repo-id username/biodreamer-protein-v1 \\
        --module protein_dreamer \\
        [--config configs/protein_dreamer/default.yaml] \\
        [--metrics results/protein_dreamer/metrics.json] \\
        [--private]

Arguments:
    --checkpoint: Path to local model checkpoint (.pt)
    --repo-id:    Hugging Face Hub repo ID (user/repo-name)
    --module:     Module name (for model card template)
    --config:     Training config YAML (included in repo)
    --metrics:    Evaluation metrics JSON (displayed in model card)
    --private:    Upload as private model (default: public)

Workflow:
    1. Load checkpoint and extract state_dict
    2. Convert to safetensors format
    3. Generate model card from template (biodreamer/hub/model_card.py)
    4. Create/update HF Hub repo
    5. Upload: safetensors, config.yaml, README.md (model card)
    6. Print Hub URL
"""
