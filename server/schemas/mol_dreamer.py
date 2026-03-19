"""
server.schemas.mol_dreamer — MolDreamer Request/Response Models.

Purpose:
    Pydantic models for MolDreamer API requests and responses.

Components to implement:
    - MolSimulationRequest:
        structure_file: UploadFile (PDB, SDF, or MOL2)
        target_property: str ('binding_affinity', 'stability', 'sasa')
        simulation_steps: int
        model_id: Optional[str]

    - MolSimulationResult:
        job_id: str
        trajectory_frames: list[CoordinateFrame]
        property_evolution: list[float]
        optimised_structure_pdb: Optional[str]

    - CoordinateFrame:
        step: int, coordinates: list[list[float]], properties: dict
"""
