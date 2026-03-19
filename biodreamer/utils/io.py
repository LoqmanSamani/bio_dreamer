"""
biodreamer.utils.io — File I/O Helpers.

Purpose:
    Centralised utilities for reading/writing common file formats used
    across the project: YAML configs, PDB structures, FASTA sequences,
    CSV data tables, and model checkpoints.

Components to implement:
    - load_config(yaml_path) → dict (parsed YAML configuration)
    - save_config(config_dict, yaml_path) → write YAML
    - load_pdb(pdb_path) → coordinates, sequence, metadata
    - save_pdb(coordinates, sequence, pdb_path) → write PDB file
    - load_fasta(fasta_path) → list of (header, sequence) tuples
    - save_fasta(sequences, fasta_path) → write FASTA file
    - load_csv(csv_path) → pandas DataFrame
    - save_results_csv(candidates, scores, csv_path) → structured results export
    - hash_sequence(sequence) → str (deterministic hash for caching)
    - ensure_dir(path) → create directory if not exists

Design notes:
    - All I/O functions handle errors gracefully with informative messages.
    - PDB I/O uses Biopython's PDB parser for robust handling.
    - Config loading supports environment variable interpolation (e.g., ${HF_TOKEN}).
"""
