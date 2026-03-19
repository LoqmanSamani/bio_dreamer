"""
evaluate.py — Model Evaluation Script.

Purpose:
    Evaluates trained world models and policies on benchmark datasets
    and reports metrics. Supports all three modules.

Usage:
    python scripts/evaluate.py \\
        --module protein_dreamer \\
        --world-model checkpoints/protein_dreamer/world_model.pt \\
        --policy checkpoints/protein_dreamer/policy/policy.pt \\
        --dataset proteingym \\
        --output-dir results/protein_dreamer/ \\
        [--metrics spearman,ndcg,mse,calibration]

Arguments:
    --module:      Module name
    --world-model: Trained world model checkpoint
    --policy:      Trained policy checkpoint (optional for world-model-only eval)
    --dataset:     Evaluation dataset name or path
    --output-dir:  Directory for evaluation results and plots
    --metrics:     Comma-separated metrics to compute

Workflow:
    1. Load world model + policy checkpoints
    2. Load evaluation dataset
    3. For each test protein/molecule/cell-state:
        a. Encode observation
        b. Dream H-step trajectory with policy
        c. Compare dreamed outcomes to ground-truth fitness/properties
    4. Compute aggregate metrics (Spearman ρ, NDCG@k, MSE, calibration)
    5. Generate visualisation plots (saved to output-dir)
    6. Print summary table and save CSV
"""
