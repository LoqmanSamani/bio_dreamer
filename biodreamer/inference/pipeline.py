"""
biodreamer.inference.pipeline — End-to-End Inference Pipeline.

Purpose:
    Orchestrates the full inference workflow from user input to ranked
    design candidates. This is what the FastAPI server calls when a user
    submits a job via the web interface.

Components to implement:
    - InferencePipeline:
        - __init__(world_model, policy, config)
        - run(input_data) → InferenceResult
            1. Parse user input (sequence, PDB, expression profile, etc.)
            2. Encode input → initial latent state z_0
            3. Run imagination rollouts (dreamer) using the policy
            4. Rank and filter candidates (candidate_ranker)
            5. Decode top candidates back to interpretable form
            6. Package results for the frontend

    - InferenceResult (dataclass):
        - candidates: list of designed sequences/perturbations
        - trajectories: list of mutation/perturbation paths
        - fitness_scores: predicted fitness for each candidate
        - uncertainties: model uncertainty for each candidate
        - visualisation_data: 3D structures, plots, heatmaps

    Domain-specific pipelines:
        - ProteinDreamerPipeline: sequence input → mutation trajectories → ranked variants
        - MolDreamerPipeline: PDB input → latent MD rollouts → optimised structures
        - CellDreamerPipeline: expression profile → perturbation plans → predicted outcomes

Design notes:
    - Pipelines must be stateless and thread-safe for concurrent server requests.
    - Support streaming results (yield partial results as imagination progresses)
      for real-time updates in the web frontend.
    - GPU memory management: load models to GPU only when needed, release after.
"""
