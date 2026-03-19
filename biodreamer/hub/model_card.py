"""
biodreamer.hub.model_card — Auto-Generate Hugging Face Model Cards.

Purpose:
    Generates standardised Hugging Face model cards (README.md) for trained
    BioDreamer models. Model cards document: architecture, training data,
    evaluation metrics, intended use, and limitations.

Components to implement:
    - generate_model_card(model_info, training_config, eval_metrics) → str (markdown)
    - ModelCardTemplate: Jinja2 template for BioDreamer model cards.

    Template sections:
        - Model description (auto-filled from model type + domain)
        - Architecture details (from training config YAML)
        - Training procedure (data, hyperparameters, compute)
        - Evaluation results (metrics from validation runs)
        - Intended use and limitations
        - How to use (code snippet)
        - Citation (bibtex for BioDreamer)

Design notes:
    - Templates live in this file as string constants or Jinja2 templates.
    - Follows Hugging Face model card guidelines: https://huggingface.co/docs/hub/model-cards
    - The web frontend displays these cards in the Model Hub page.
    - Automatically called during upload_model() in hub/upload.py.
"""
