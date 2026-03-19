"""
biodreamer.inference.candidate_ranker — Candidate Ranking and Filtering.

Purpose:
    After dreaming N trajectories, this module ranks the final candidates
    and selects a diverse, high-quality shortlist for the user.

Components to implement:
    - CandidateRanker:
        - rank(dream_result, top_k) → ranked list of candidates
        - filter_by_fitness(candidates, min_fitness) → filtered candidates
        - filter_by_novelty(candidates, wild_type, min_distance) → filter duplicates
        - diversify(candidates, n_select) → diverse subset (MaxMin or DPP selection)

    Ranking criteria:
        - Predicted fitness (primary)
        - Model confidence (prefer low-uncertainty candidates)
        - Novelty (distance from wild-type and from each other)
        - Synthesisability (number of mutations — fewer is easier)
        - Pareto optimality (for multi-objective cases)

Design notes:
    - The ranker's output is what the user sees in the web frontend's results
      table: sequence, mutations, predicted fitness, confidence, diversity score.
    - DPP (Determinantal Point Process) selection ensures the top-K candidates
      are spread across the fitness landscape, not clustered in one region.
    - For active learning, the ranker interfaces with active_learning.py to
      select candidates for experimental validation.
"""
