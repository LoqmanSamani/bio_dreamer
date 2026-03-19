"""
biodreamer.inference — Inference Pipelines.

Modules for running trained models in production (web server) and research
(notebooks, scripts). Optimised for latency and throughput.

Modules:
    - pipeline:          End-to-end inference pipeline (input → encode → dream → rank → output)
    - dreamer:           Imagination rollout engine (dream mutation/perturbation trajectories)
    - candidate_ranker:  Rank and filter designed candidates by predicted fitness + diversity
"""
