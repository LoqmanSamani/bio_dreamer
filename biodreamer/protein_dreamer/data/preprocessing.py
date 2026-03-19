"""
biodreamer.protein_dreamer.data.preprocessing — Protein Data Preprocessing.

Purpose:
    Utilities for converting raw protein data (sequences, structures, DMS
    measurements) into tensors ready for model training.

Components to implement:
    - tokenise_sequence(seq_string) → token IDs (using ESM-2 tokeniser)
    - parse_mutation_string(mut_str) → (position, wt_aa, mut_aa, edit_type)
    - encode_mutation(mutation, max_seq_len) → mutation embedding tensor
    - load_structure(pdb_path_or_sequence) → backbone coordinates (L × 4 × 3)
    - predict_structure_esmfold(sequence) → (coordinates, plddt_scores)
    - build_protein_graph(coordinates) → PyG Data (residue-level graph for GVP-GNN)
    - compute_distance_map(coordinates) → (L × L) Cα distance matrix
    - normalise_fitness(raw_scores, method='quantile') → [0,1] normalised fitness
    - augment_dms_data(dataset, strategy) → augmented dataset (reverse mutations, shuffles)

Design notes:
    - Structure prediction (ESMFold) is expensive — cache predicted structures
      keyed by sequence hash.
    - The ESM-2 tokeniser and GVP-GNN graph builder will be reused across
      training, inference, and the web server. Implement once here.
    - Support batch preprocessing for large DMS assays (100k+ variants).
"""
