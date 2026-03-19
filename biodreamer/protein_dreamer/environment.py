"""
biodreamer.protein_dreamer.environment — Protein Fitness Landscape Environment.

Purpose:
    Wraps the protein fitness landscape as a gym-like environment for RL
    training and evaluation. Provides the "real" environment that the
    world model learns to approximate.

Components to implement:
    - ProteinEnvironment:
        - __init__(wild_type_sequence, fitness_oracle, structure_oracle)
        - reset() → initial state (sequence, structure, fitness)
        - step(mutation) → (next_state, reward, done, info)
        - render() → sequence alignment + structure visualisation data

    Fitness oracle options:
        1. DMSLookupOracle: Exact fitness lookup from a DMS dataset.
           Used for benchmarking on known fitness landscapes (ProteinGym).
        2. ESMFoldOracle: Uses ESMFold pLDDT + ProteinMPNN inverse-folding
           score as a proxy for designability. Fast, no wet lab needed.
        3. PredictorOracle: Uses a pre-trained fitness predictor (Tranception,
           ESM-2 zero-shot, or a fine-tuned model) as the fitness function.
        4. WetLabOracle (interface): Placeholder for real experimental
           validation — accepts fitness measurements from external sources.

    Structure oracle options:
        - ESMFold (fast, single-sequence)
        - AlphaFold2 / ColabFold (slower, more accurate)

Design notes:
    - During world model training, the environment provides ground-truth
      transitions from DMS data.
    - During policy training, the agent does NOT use the environment —
      it trains entirely in the world model (dreaming).
    - The environment is used only for final evaluation and active learning.
"""
