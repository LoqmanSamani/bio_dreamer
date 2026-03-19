"""
biodreamer.protein_dreamer.data.dataset — Protein Fitness Landscape Datasets.

Purpose:
    PyTorch Dataset classes for loading protein fitness data (deep mutational
    scanning assays, stability measurements) as training data for the world
    model and reward head.

Components to implement:
    - ProteinGymDataset(torch.utils.data.Dataset):
        Loads ProteinGym benchmark data (200+ DMS assays).
        Each sample: (wild_type_seq, mutation_list, fitness_score, assay_id)
        Supports: single-mutant substitutions, multi-mutant combinations, indels.
        Reference: Notin et al. (NeurIPS 2023 Datasets & Benchmarks)

    - TsuboyamaDataset(torch.utils.data.Dataset):
        Loads the Tsuboyama mega-scale ΔΔG stability dataset (500k+ measurements).
        Each sample: (wild_type_seq, mutation, ΔΔG_value, domain_id)
        Reference: Tsuboyama et al. (Nature 2023)

    - FitnessTransitionDataset:
        Constructs (state, action, next_state, reward) tuples from DMS data
        for world model training. Pairs mutations into sequential transitions:
        e.g., WT → single_mutant → double_mutant.

    - CustomAssayDataset:
        For user-provided fitness data (CSV format: sequence, fitness).
        Used in the web interface's "Upload your data" workflow.

Design notes:
    - ProteinGym and Tsuboyama data should be downloaded once and cached locally
      (scripts/download_data.sh handles this).
    - Multi-mutant transition construction is non-trivial: need to identify paths
      in the mutation graph where intermediate measurements exist.
    - Split strategies: by protein (for generalisation evaluation) or by
      mutation depth (train on singles, test on doubles/triples).
"""
