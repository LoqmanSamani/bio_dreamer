"""
biodreamer.training.active_learning — Active Learning Loop.

Purpose:
    Implements the closed-loop active learning cycle that bridges the
    computational world model with real experimental validation:

        Dream → Propose candidates → Evaluate (oracle or wet lab) → Update model → Repeat

    This is how ProteinDreamer integrates with real protein engineering campaigns.

Components to implement:
    - ActiveLearningLoop:
        - step(budget) → list of proposed candidates ranked by acquisition score
        - update(new_measurements) → retrain world model on expanded dataset
        - report() → summary of active learning progress (rounds, improvement)

    Acquisition functions (decide which candidates to evaluate):
        - MaxUncertainty: propose the mutants the world model is least certain about
        - MaxEFE: propose mutants with highest Expected Free Energy (exploration)
        - MaxPredictedFitness: propose the top-ranked candidates (exploitation)
        - UCB (Upper Confidence Bound): balance mean prediction + uncertainty
        - BatchDiversity: select a diverse batch that covers different regions
          of the fitness landscape (DPP or clustering-based)

Design notes:
    - In the web frontend, users can trigger active learning rounds:
      world model proposes candidates → user reviews → marks which ones were
      tested → uploads results → world model fine-tunes.
    - For ProteinGym benchmarking: simulate the active learning loop by
      revealing DMS measurements one batch at a time and measuring how
      quickly the world model discovers the best variants.
    - Budget-constrained: each round evaluates at most B candidates (B is
      set by the user or the experimental cost budget).
"""
