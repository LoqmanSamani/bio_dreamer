"""
test_protein_encoder.py — Unit Tests for biodreamer.protein_dreamer.encoder.

Purpose:
    Tests the dual-branch ProteinEncoder (ESM-2 sequence + GVP-GNN structure).

Tests to implement:
    - test_sequence_branch_output: ESM-2 branch produces per-residue embeddings
    - test_structure_branch_output: GVP-GNN branch produces node features
    - test_joint_embedding: combined output has correct dimensions
    - test_lora_finetuning_mode: LoRA adapters are properly attached
    - test_frozen_backbone: base ESM-2 weights don't update when frozen
    - test_variable_length_sequences: handles sequences of different lengths
"""
