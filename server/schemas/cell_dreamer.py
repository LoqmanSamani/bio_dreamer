"""
server.schemas.cell_dreamer — CellDreamer Request/Response Models.

Purpose:
    Pydantic models for CellDreamer API requests and responses.

Components to implement:
    - PerturbationPlanRequest:
        expression_file: UploadFile (CSV or h5ad)
        target_cell_type: Optional[str]
        target_markers: Optional[dict] (gene → 'up'/'down')
        perturbation_budget: int (max perturbations)
        available_perturbations: Optional[list[str]] (e.g., gene names for CRISPR)
        model_id: Optional[str]

    - PerturbationPlanResult:
        job_id: str
        perturbation_sequence: list[PerturbationStep]
        predicted_trajectory: list[CellState]
        umap_coordinates: list[list[float]]
        top_differential_genes: list[GeneEffect]

    - PerturbationStep:
        step: int, perturbation_type: str, target_gene: str, dose: Optional[float]

    - CellState:
        step: int, expression_profile: dict[str, float], distance_to_target: float

    - GeneEffect:
        gene_name: str, log_fold_change: float, p_value: float
"""
