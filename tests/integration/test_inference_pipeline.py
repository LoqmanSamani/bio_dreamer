"""
test_inference_pipeline.py — Integration Tests for Inference Pipelines.

Purpose:
    Tests the full inference flow: input parsing → encoding → dreaming →
    ranking → output formatting. Uses pre-trained dummy checkpoints.

Tests to implement:
    - test_protein_dreamer_pipeline: sequence in → ranked candidates out
    - test_mol_dreamer_pipeline: structure in → trajectory frames out
    - test_cell_dreamer_pipeline: perturbation request in → ranked plans out
    - test_dream_with_branching: branching rollout produces diverse candidates
    - test_candidate_ranking: candidates are sorted by fitness, diversity applied
    - test_streaming_output: pipeline yields intermediate results
    - test_pipeline_from_config: loads pipeline from YAML config
"""
