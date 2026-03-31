# BioDreamer — Project Literature Overview

> Papers covering the **full scope** of the BioDreamer programme: JEPA world models, model-based RL, active inference, generative models for biology, ML for molecular dynamics, protein design, and cellular perturbation modelling.
> This is a curated reading list for writing the **blog post introduction** to BioDreamer — broad enough to motivate each pillar, concise enough to not overwhelm. For the deep ProteinDreamer-specific dive, see `protein_dreamer/pro_dream_lits.md`.

**Importance levels:** 🔴 Core — central to the blog narrative, likely cited. 🟡 Supporting — provides context for key claims. 🟢 Background — good to know, may or may not be referenced.

---

## 1. World Models & JEPA (Backbone Architecture)

BioDreamer's unified architecture is built on **JEPA (Joint-Embedding Predictive Architecture)** — encode observations into latent states, predict transitions in latent space via an energy-based objective, and plan with RL on imagined trajectories. No decoder needed during planning. These papers define the paradigm.

### 🔴 1.1 Causal-JEPA — Learning World Models through Object-Level Latent Interventions

- **Authors:** Nam, Le Lidec, Maes, LeCun, Balestriero (AMI Labs)
- **Venue:** arXiv 2026
- **Link:** https://arxiv.org/abs/2602.11389
- **Why it matters:** The **primary architectural blueprint** for BioDreamer. Extends JEPA with causal reasoning via latent-space interventions — predicting how actions change the world state entirely in latent space, without reconstruction. In BioDreamer, biological interventions (mutations, perturbations, force-field changes) are exactly these latent interventions. The JEPA energy $E_\theta(z_t, a_t, z_{t+1})$ also connects directly to the Free Energy Principle — both minimise prediction error.

### 🔴 1.2 LeWorldModel — Stable End-to-End JEPA from Pixels

- **Authors:** Maes, Le Lidec, Scieur, LeCun, Balestriero (AMI Labs)
- **Venue:** arXiv 2026
- **Link:** https://arxiv.org/abs/2603.19312
- **Why it matters:** Solves the training stability problem for JEPA world models — EMA target encoder, SIGReg regularisation (enforces $z_t \sim \mathcal{N}(0, I)$ with a single hyperparameter $\lambda$). Provides the practical training recipe that BioDreamer adapts across all three pillars. Also demonstrates empirically that latent-space prediction outperforms reconstruction-based approaches (e.g., DreamerV3's RSSM).

### 🔴 1.3 DIAMOND — Diffusion for World Modeling

- **Authors:** Alonso, Jelley, Micheli, Fleuret
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.12399
- **Why it matters:** The key precedent for BioDreamer's **Latent Diffusion JEPA** variant (Architecture A). Demonstrates that diffusion-based dynamics produce more accurate long-horizon rollouts than deterministic models and capture multi-modal transitions. DIAMOND operates in observation space (pixels); BioDreamer moves the diffusion into JEPA latent space for massive efficiency gains — analogous to Stable Diffusion's move from pixel-space to latent-space.

### 🟡 1.4 DreamerV3 — Mastering Diverse Domains Through World Models

- **Authors:** Hafner, Pasukonis, Ba, Lillicrap
- **Venue:** arXiv 2023 / Nature 2025
- **Link:** https://arxiv.org/abs/2301.04104
- **Why it matters:** The flagship RSSM-based world-model RL agent. Historically, BioDreamer's original inspiration — it demonstrated that a single world-model architecture can master 150+ diverse environments. BioDreamer retains DreamerV3's core principles (imagination-based policy training, symlog predictions) but replaces the RSSM backbone with JEPA: no decoder during planning, energy-based objective connecting to Active Inference, and intervention-aware prediction via Causal-JEPA. Serves as a **historical baseline** in the library.

### 🟢 1.5 Ha & Schmidhuber — World Models (Original)

- **Authors:** Ha, Schmidhuber
- **Venue:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1803.10122
- **Why it matters:** The seminal paper that coined "world models" in the modern deep RL sense. VAE encoder → MDN-RNN dynamics → compact controller. Sets up the conceptual framework that BioDreamer inherits: learn to dream, then act in the dream.

---

## 2. Active Inference & Free Energy Principle (Theoretical Backbone)

BioDreamer's exploration–exploitation strategy is grounded in Active Inference. The agent selects actions (mutations, perturbations) that minimise *expected free energy* — simultaneously seeking high reward (exploitation) and reducing model uncertainty (exploration). The JEPA energy objective is mathematically analogous to variational free energy.

### 🔴 2.1 Active Inference: The Free Energy Principle in Mind, Brain, and Behavior

- **Authors:** Parr, Pezzulo, Friston
- **Venue:** MIT Press 2022
- **Link:** https://doi.org/10.7551/mitpress/12441.001.0001
- **Why it matters:** The definitive textbook. Covers expected free energy (EFE) decomposition into pragmatic value (reward) + epistemic value (information gain). BioDreamer adapts this framework: the world model is the generative model, mutations/perturbations are policies, and EFE drives action selection across all three pillars.

### 🟡 2.2 Reinforcement Learning Through Active Inference

- **Authors:** Tschantz, Millidge, Seth, Buckley
- **Venue:** arXiv 2020
- **Link:** https://arxiv.org/abs/2002.12636
- **Why it matters:** The technical bridge between Active Inference and deep RL. Shows how EFE minimisation subsumes reward maximisation + information gain. Provides the recipe for implementing active inference with neural networks — directly applicable to BioDreamer's policy.

### 🟢 2.3 Active Inference: Demystified and Compared

- **Authors:** Sajid, Ball, Parr, Friston
- **Venue:** Neural Computation 2021
- **Link:** https://doi.org/10.1162/neco_a_01357
- **Why it matters:** Clear comparison of active inference with standard RL, Bayesian RL, and control-as-inference. Useful for positioning BioDreamer's theoretical contribution relative to conventional RL approaches.

---

## 3. Protein Foundation Models & Structure Prediction

The encoders and cheap oracles that power ProteinDreamer (and inform MolDreamer's structural understanding).

### 🔴 3.1 ESM-2 — Language Models of Protein Evolution

- **Authors:** Lin, Akin, Rao, ..., Rives
- **Venue:** Science 2023
- **Link:** https://doi.org/10.1126/science.ade2574
- **Why it matters:** ProteinDreamer's **primary sequence encoder**. ESM-2 embeddings encode evolutionary and structural information; zero-shot fitness prediction via masked marginals is a natural baseline. Also introduces ESMFold for fast structure prediction — used as a cheap oracle in the active learning loop.

### 🟡 3.2 AlphaFold2 — Highly Accurate Protein Structure Prediction

- **Authors:** Jumper, Evans, Pritzel, ..., Hassabis
- **Venue:** Nature 2021
- **Link:** https://doi.org/10.1038/s41586-021-03819-2
- **Why it matters:** Revolutionised structural biology. Predicted structures serve as cheap stand-ins for experimental structures when training ProteinDreamer's world model and validating designed proteins.

### 🟡 3.3 ESM-3 — Simulating 500 Million Years of Evolution with a Language Model

- **Authors:** Hayes, Mahber, ..., Rives (EvolutionaryScale)
- **Venue:** bioRxiv 2024
- **Link:** https://doi.org/10.1101/2024.07.01.600583
- **Why it matters:** Multi-modal protein foundation model (sequence + structure + function tokens). Represents the frontier of *one-shot* generative protein design — positioning ProteinDreamer's iterative, world-model-guided approach as complementary rather than competing.

---

## 4. Generative Protein Design (One-Shot Baselines)

These are the state-of-the-art methods that ProteinDreamer improves upon by adding iterative, RL-guided planning. The blog post uses these to set up the "gap" BioDreamer fills: powerful generation, but no multi-step planning.

### 🔴 4.1 ProteinMPNN — Robust Deep Learning–Based Protein Sequence Design

- **Authors:** Dauparas, ..., Baker
- **Venue:** Science 2022
- **Link:** https://doi.org/10.1126/science.add2187
- **Why it matters:** The standard inverse-folding method. One-shot: given a backbone → design a sequence. No iterative refinement. ProteinDreamer could use ProteinMPNN as a sub-module or as a baseline to beat.

### 🔴 4.2 RFdiffusion — De Novo Protein Design with Diffusion Models

- **Authors:** Watson, ..., Baker
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06415-8
- **Why it matters:** Generative diffusion model for novel protein backbones — binders, symmetric assemblies, motif scaffolds. Powerful one-shot design, but no multi-step fitness optimisation loop. Key baseline.

### 🟢 4.3 EvoDiff — Protein Generation via Sequence Diffusion

- **Authors:** Alamdari, ..., Gitter
- **Venue:** Nature Biotechnology 2023
- **Link:** https://doi.org/10.1038/s41587-023-01902-9
- **Why it matters:** Diffusion directly in sequence space (discrete diffusion). Shows structure isn't required for generative protein design.

---

## 5. RL for Biological Sequence Design

The closest existing work to ProteinDreamer. The blog post references these to show that RL-for-proteins is emerging, but **no one has combined JEPA world models + Active Inference for iterative protein design**.

### 🔴 5.1 Accelerating Protein Engineering with Fitness Landscape Modelling and RL (μFormer + μSearch)

- **Authors:** Sun, He, Deng, Liu, Zhao, Jiang, ...
- **Venue:** Nature Machine Intelligence 2025
- **Link:** https://doi.org/10.1038/s42256-025-01103-w
- **Why it matters:** Most recent high-profile work combining fitness predictors with RL. Uses a deep learning fitness predictor + RL-based search — but no world model with latent dynamics, no active inference, no dreaming. Key paper to differentiate from.

### 🟡 5.2 EvoPlay — Self-Play RL Guides Protein Engineering

- **Authors:** Wang, Tang, Huang, ..., Yang
- **Venue:** Nature Machine Intelligence 2023
- **Link:** https://doi.org/10.1038/s42256-023-00691-9
- **Why it matters:** MCTS + self-play for navigating protein fitness landscapes. Closest in spirit to ProteinDreamer, but uses a discriminative fitness oracle — not a generative world model with latent state transitions.

### 🟡 5.3 Biological Sequence Design with GFlowNets

- **Authors:** Jain, Bengio, Hernandez-Garcia, ..., Bengio
- **Venue:** ICML 2022
- **Link:** https://proceedings.mlr.press/v162/jain22a.html
- **Why it matters:** Diversity-seeking alternative to standard RL for sequence design. Compares with DynaPPO (model-based RL). GFlowNets could serve as a policy within BioDreamer's framework — the world model provides the environment, GFlowNets or actor-critic provide the policy.

---

## 6. MolDreamer Foundations (Molecular Dynamics ML)

### 🟡 6.1 MACE — Higher Order Equivariant Message Passing for Force Fields

- **Authors:** Batatia, Kovacs, Simm, Ortner, Csányi
- **Venue:** NeurIPS 2022
- **Link:** https://proceedings.neurips.cc/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
- **Why it matters:** State-of-the-art equivariant GNN for interatomic potentials. Candidate architecture for MolDreamer's SE(3)-equivariant encoder.

### 🟢 6.2 Timewarp — Transferable Acceleration of Molecular Dynamics

- **Authors:** Klein, Krämer, Noé
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2302.01170
- **Why it matters:** Learns large jumps in MD trajectory space using normalising flows — directly analogous to MolDreamer's goal of variable-length latent dynamics. Shows that learned time-stepping for MD is achievable.

---

## 7. CellDreamer Foundations (Cellular Perturbation Models)

### 🟡 7.1 GEARS — Predicting Transcriptional Outcomes of Novel Multigene Perturbations

- **Authors:** Roohani, Huang, Leskovec
- **Venue:** Nature Biotechnology 2024
- **Link:** https://doi.org/10.1038/s41587-023-01905-6
- **Why it matters:** The closest existing work to CellDreamer — predicts gene expression changes under perturbations. But limited to **steady-state** predictions (no dynamics, no RL planning). Key baseline CellDreamer aims to surpass.

### 🟢 7.2 Replogle et al. — Genome-Wide Perturb-seq

- **Authors:** Replogle, Saunders, Pogson, ..., Weissman
- **Venue:** Cell 2022
- **Link:** https://doi.org/10.1016/j.cell.2022.05.013
- **Why it matters:** The primary **training dataset** for CellDreamer. Genome-scale CRISPRi Perturb-seq — ~2.5M single-cell profiles across ~10,000 gene knockdowns.

---

## 8. Fitness Landscapes & Key Data Sources

### 🔴 8.1 ProteinGym — Large-Scale Benchmarks for Protein Fitness Prediction and Design

- **Authors:** Notin, Kollasch, Ritter, ..., Marks
- **Venue:** NeurIPS 2023 Datasets & Benchmarks
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html
- **Why it matters:** **The primary benchmark** for ProteinDreamer. 217 substitution DMS assays covering diverse proteins. Any claim about ProteinDreamer's effectiveness must be evaluated here.

### 🟡 8.2 Mega-Scale Stability Data — Tsuboyama et al.

- **Authors:** Tsuboyama, Dauparas, Chen, ..., Marks, Hartl, et al.
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06328-6
- **Why it matters:** >500,000 ΔΔG measurements across ~100 protein domains. The richest single training data source for ProteinDreamer's world model and reward head.

---

## 9. Reviews (Big-Picture Framing)

### 🔴 9.1 Machine Learning for Functional Protein Design

- **Authors:** Notin, Rollins, Gal, Sander, Marks
- **Venue:** Nature Biotechnology 2024
- **Link:** https://doi.org/10.1038/s41587-024-02127-0
- **Why it matters:** Comprehensive state-of-the-field review. Maps all ML approaches to protein design and identifies iterative optimisation as under-explored — exactly the gap BioDreamer fills. Essential for framing the blog post's "why now" argument.

### 🟡 9.2 Adaptive Machine Learning for Protein Engineering

- **Authors:** Hie, Yang
- **Venue:** Current Opinion in Structural Biology 2022
- **Link:** https://doi.org/10.1016/j.sbi.2021.11.002
- **Why it matters:** Reviews the ML-guided protein engineering loop: train surrogate → propose → test → retrain. The community already does iterative design — BioDreamer formalises this with world models + Active Inference.

---

## Summary: Suggested Reading Order for Blog Writing

1. **JEPA world models (the backbone):** Causal-JEPA → LeWorldModel → DIAMOND → DreamerV3 (historical context) → Ha & Schmidhuber (conceptual origin)
2. **Active Inference (the theory):** Parr et al. textbook → Tschantz et al. (RL through AI)
3. **Why BioDreamer is needed (the gap):** Notin et al. review (2024) → ProteinMPNN → RFdiffusion → ESM-3 (one-shot methods) → μFormer+μSearch → EvoPlay (RL attempts, but no world model)
4. **Encoders & oracles:** ESM-2 → AlphaFold2
5. **Data:** ProteinGym → Tsuboyama mega-scale
6. **Module-specific context:** MACE (MolDreamer) → GEARS (CellDreamer)
