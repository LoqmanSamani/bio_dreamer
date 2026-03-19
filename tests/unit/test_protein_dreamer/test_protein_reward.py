"""
test_protein_reward.py — Unit Tests for biodreamer.protein_dreamer.reward.

Tests to implement:
    - test_single_objective: ΔΔG prediction head produces scalar output
    - test_multi_objective: multiple heads produce individual + scalarised output
    - test_pareto_scalarisation: Tchebycheff and weighted-sum modes
    - test_reward_normalisation: outputs are properly scaled
"""
