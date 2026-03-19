"""
test_encoder.py — Unit Tests for biodreamer.core.encoder.

Purpose:
    Tests the BaseEncoder interface and ensures concrete implementations
    (MolEncoder, ProteinEncoder, CellEncoder) conform to the contract.

Tests to implement:
    - test_base_encoder_is_abstract: cannot instantiate BaseEncoder directly
    - test_encoder_output_shape: encode(observation) → z_t with correct dims
    - test_encoder_deterministic: same input produces same output (no sampling)
    - test_encoder_batch_dimension: handles batch of observations correctly
    - test_encoder_device_transfer: works on both CPU and CUDA
"""
