"""
test_mol_encoder.py — Unit Tests for biodreamer.mol_dreamer.encoder.

Tests to implement:
    - test_equivariant_output: SE(3)-equivariant GNN respects rotation invariance
    - test_graph_construction: atomic graph from PDB has correct nodes/edges
    - test_encoder_output_shape: latent z_t has expected dimensions
    - test_different_architectures: EGNN, PaiNN, SchNet variants all produce valid output
"""
