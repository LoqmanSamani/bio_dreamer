"""
server.schemas.protein_dreamer — ProteinDreamer Request/Response Models.

Purpose:
    Pydantic models for validating ProteinDreamer API requests and
    serialising responses.

Components to implement:
    - ProteinDesignRequest:
        wild_type_sequence: str (amino acid sequence)
        pdb_file: Optional[UploadFile] (3D structure)
        objective: str ('stability', 'affinity', 'activity', 'multi')
        objective_weights: Optional[dict] (for multi-objective)
        mutation_budget: int (max mutations per trajectory)
        n_trajectories: int (how many dream trajectories to generate)
        imagination_horizon: int (max steps per trajectory)
        model_id: Optional[str] (which world model to use)

    - ProteinDesignResult:
        job_id: str
        candidates: list[CandidateVariant]
        trajectories: list[MutationTrajectory]
        wild_type_fitness: float
        best_predicted_fitness: float

    - CandidateVariant:
        sequence: str, mutations: list[str], predicted_fitness: dict,
        confidence: float, structure_pdb: Optional[str]

    - MutationTrajectory:
        steps: list[MutationStep] (sequence of mutations with intermediate fitness)
"""
