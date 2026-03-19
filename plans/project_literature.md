# BioDreamer — Project Literature Overview

> Papers covering the **full scope** of the BioDreamer programme: world models, model-based RL, generative models for biology, ML for molecular dynamics, protein design, and cellular perturbation modelling.
> This is a broad-strokes reading list — for the deep ProteinDreamer-specific dive, see `pro_dream_lits.md`.

---

## 1. World Models & Model-Based RL (Foundational)

These papers define the "world model" paradigm that BioDreamer transfers to biological domains. Understanding their architectures (RSSM, discrete tokenisation, diffusion dynamics) is essential for designing the biological counterparts.

### 1.1 DreamerV3 — Mastering Diverse Domains Through World Models
- **Authors:** Hafner, Pasukonis, Ba, Lillicrap
- **Venue:** arXiv 2023 / Nature 2025
- **Link:** https://arxiv.org/abs/2301.04104
- **Why study:** The flagship world-model RL agent. Introduces the Recurrent State-Space Model (RSSM) with symlog predictions and a general recipe for training latent dynamics models + actor-critic policies across wildly different environments. **This is the direct architectural inspiration for all three BioDreamer pillars.** Must understand: RSSM latent structure, KL balancing, reward/value prediction heads, imagination-based policy optimisation.

### 1.2 IRIS — Transforming the World Model with Transformers
- **Authors:** Micheli, Alonso, Fleuret
- **Venue:** ICML 2023
- **Link:** https://arxiv.org/abs/2209.00588
- **Why study:** Replaces the RSSM with a discrete autoencoder + autoregressive transformer world model. Shows that tokenised world models can match or beat continuous-latent ones. Relevant for ProteinDreamer where protein sequences are naturally discrete tokens.

### 1.3 DIAMOND — Diffusion for World Modeling
- **Authors:** Alonso, Jelley, Micheli, Fleuret
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.12399
- **Why study:** Uses a **diffusion model** as the world model's dynamics backbone. Directly relevant to MolWorld (where molecular dynamics are stochastic) and ProteinDreamer (where structure changes can be modelled as denoising). Demonstrates that diffusion world models produce more accurate long-horizon rollouts than RSSM-based ones.

### 1.4 Ha & Schmidhuber — World Models (Original)
- **Authors:** Ha, Schmidhuber
- **Venue:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1803.10122
- **Why study:** The seminal paper that coined "world models" in the modern deep RL sense. VAE encoder + MDN-RNN dynamics + compact controller. Sets up the conceptual framework: learn to dream, then act in the dream.

---

## 2. Protein Structure Prediction & Foundation Models

These are the tools that provide the "cheap oracles" and encoders for ProteinDreamer and MolWorld.

### 2.1 AlphaFold2 — Highly Accurate Protein Structure Prediction
- **Authors:** Jumper, Evans, Pritzel, ..., Hassabis
- **Venue:** Nature 2021
- **Link:** https://doi.org/10.1038/s41586-021-03819-2
- **Why study:** Revolutionised structural biology. Its predicted structures serve as cheap stand-ins for experimental structures when training ProteinDreamer's world model. Also key for evaluating whether designed proteins fold correctly.

### 2.2 ESM-2 — Language Models of Protein Evolution
- **Authors:** Lin, Akin, Rao, ..., Rives
- **Venue:** Science 2023
- **Link:** https://doi.org/10.1126/science.ade2574
- **Why study:** State-of-the-art protein language model. ESM-2 embeddings are the **primary sequence encoder** for ProteinDreamer. Its zero-shot fitness prediction capabilities (via masked marginals) are a natural baseline. Also introduces ESMFold for fast structure prediction.

### 2.3 ESM-3 — Simulating 500 Million Years of Evolution with a Language Model
- **Authors:** Hayes, Mahber, ..., Rives (EvolutionaryScale)
- **Venue:** bioRxiv 2024
- **Link:** https://doi.org/10.1101/2024.07.01.600583
- **Why study:** Multi-modal protein language model (sequence + structure + function tokens). Represents the frontier of one-shot generative protein design — understanding its capabilities and limitations helps position ProteinDreamer's iterative approach as complementary.

---

## 3. Generative Protein Design (One-Shot Methods — Baselines)

These are the current state-of-the-art methods that ProteinDreamer aims to improve upon by adding iterative, RL-guided planning.

### 3.1 ProteinMPNN — Robust Deep Learning–Based Protein Sequence Design
- **Authors:** Dauparas, ..., Baker
- **Venue:** Science 2022
- **Link:** https://doi.org/10.1126/science.add2187
- **Why study:** The standard inverse-folding method: given a backbone structure, design a sequence that folds to it. One-shot, no iterative refinement. ProteinDreamer could use ProteinMPNN as a sub-module (sequence proposal given a dreamed structure) or as a baseline to beat.

### 3.2 RFdiffusion — De Novo Protein Design with Diffusion Models
- **Authors:** Watson, ..., Baker
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06415-8
- **Why study:** Generative diffusion model for protein backbone structures. Can design novel folds, binders, and symmetric assemblies. One-shot again — no multi-step planning or fitness optimisation loop. Key baseline and a potential component (backbone generator) within ProteinDreamer.

### 3.3 Chroma — Generative Protein Design by Direct Structure Prediction
- **Authors:** Ingraham, ..., Stanton (Generate Biomedicines)
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06728-8
- **Why study:** Alternative diffusion approach to protein structure generation, with conditioning on various properties. Demonstrates that property-guided generation is feasible — but still one-shot.

### 3.4 EvoDiff — Protein Generation via Sequence Diffusion
- **Authors:** Alamdari, ..., Gitter
- **Venue:** Nature Biotechnology 2023
- **Link:** https://doi.org/10.1038/s41587-023-01902-9
- **Why study:** Diffusion models operating directly in sequence space (discrete diffusion). Shows that you don't need structure for generative protein design. Relevant architecture for ProteinDreamer's sequence-space world model variants.

---

## 4. ML for Molecular Dynamics (MolWorld Foundations)

### 4.1 MACE — Higher Order Equivariant Message Passing Neural Networks for Force Fields
- **Authors:** Batatia, Kovacs, Simm, Ortner, Csányi
- **Venue:** NeurIPS 2022
- **Link:** https://proceedings.neurips.cc/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html
- **Why study:** State-of-the-art equivariant GNN for learning interatomic potentials. Its architecture (many-body equivariant messages) is a candidate for MolWorld's encoder. Demonstrates that SE(3)-equivariant networks can replace expensive quantum-mechanical calculations.

### 4.2 NequIP — E(3)-Equivariant Graph Neural Networks for Data-Efficient Atomistic Potentials
- **Authors:** Batzner, Musaelian, Sun, ..., Kozinsky
- **Venue:** Nature Communications 2022
- **Link:** https://doi.org/10.1038/s41467-022-29939-5
- **Why study:** Pioneered equivariant neural network interatomic potentials with remarkable data efficiency. Key architecture reference for MolWorld's encoder. Shows that E(3)-equivariance is critical for molecular systems.

### 4.3 TorchMD-NET — Equivariant Transformers for Neural Network Potentials
- **Authors:** Thölke, De Fabritiis
- **Venue:** ICLR 2022
- **Link:** https://arxiv.org/abs/2202.02541
- **Why study:** Combines transformers with equivariant features for molecular potentials. Integrates with the TorchMD ecosystem for running ML-driven MD. Relevant as an alternative encoder architecture for MolWorld.

### 4.4 Timewarp — Transferable Acceleration of Molecular Dynamics
- **Authors:** Klein, Krämer, Noé
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2302.01170
- **Why study:** Learns to make large jumps in MD trajectory space using normalizing flows — directly analogous to MolWorld's goal of variable-length latent dynamics. Shows that learned time-stepping for MD is achievable.

---

## 5. Cellular Perturbation Models (CellDreamer Foundations)

### 5.1 GEARS — Predicting Transcriptional Outcomes of Novel Multigene Perturbations
- **Authors:** Roohani, Huang, Leskovec
- **Venue:** Nature Biotechnology 2024
- **Link:** https://doi.org/10.1038/s41587-023-01905-6
- **Why study:** Graph-based deep learning model for predicting gene expression changes under single and combinatorial perturbations. The closest existing work to CellDreamer, but limited to **steady-state** predictions (no dynamics, no RL planning). Key baseline.

### 5.2 CPA — Compositional Perturbation Autoencoder
- **Authors:** Lotfollahi, Klimovskaia, De Donno, ..., Theis
- **Venue:** NeurIPS 2021 / Molecular Systems Biology 2023
- **Link:** https://doi.org/10.15252/msb.202211517
- **Why study:** VAE-based model that disentangles drug/genetic perturbation effects from cell-type effects. Enables combinatorial perturbation prediction. Key baseline for CellDreamer; its latent space could serve as the starting point for CellDreamer's dynamics model.

### 5.3 scGPT — Foundation Model for Single-Cell Multi-Omics
- **Authors:** Cui, Wang, Maan, ..., Wang
- **Venue:** Nature Methods 2024
- **Link:** https://doi.org/10.1038/s41592-024-02201-0
- **Why study:** Transformer foundation model pre-trained on 33M+ cells. Can do gene perturbation prediction, cell type annotation, and gene network inference. Represents the frontier of single-cell foundation models. CellDreamer's encoder could leverage scGPT embeddings.

### 5.4 Replogle et al. — Genome-Wide Perturb-seq
- **Authors:** Replogle, Saunders, Pogson, ..., Weissman
- **Venue:** Cell 2022
- **Link:** https://doi.org/10.1016/j.cell.2022.05.013
- **Why study:** The primary **training dataset** for CellDreamer. Genome-scale CRISPRi Perturb-seq in K562 and RPE1 cells — ~2.5M single-cell profiles across ~10,000 gene knockdowns. Understanding this dataset's structure and limitations is essential.

### 5.5 Norman et al. — Exploring Genetic Interaction Manifold with Combinatorial CRISPRa
- **Authors:** Norman, Horlbeck, Replogle, ..., Weissman
- **Venue:** Science 2019
- **Link:** https://doi.org/10.1126/science.aax4438
- **Why study:** Combinatorial CRISPRa perturbation dataset — the standard benchmark for evaluating multi-gene perturbation prediction models (used by GEARS, CPA, scGPT). Critical for evaluating CellDreamer.

---

## 6. RL for Biological Sequence Design

These papers directly attack the problem of using RL to design biological sequences — the closest existing work to ProteinDreamer. Understanding their strengths and gaps positions ProteinDreamer's novelty.

### 6.1 Biological Sequence Design with GFlowNets
- **Authors:** Jain, Bengio, Hernandez-Garcia, ..., Bengio
- **Venue:** ICML 2022
- **Link:** https://proceedings.mlr.press/v162/jain22a.html
- **Why study:** Introduces Generative Flow Networks for biological sequence design — a diversity-seeking alternative to standard RL. Compares with DynaPPO (model-based RL for sequences). Directly relevant as a competing method; GFlowNets could also be used as ProteinDreamer's policy.

### 6.2 EvoPlay — Self-Play RL Guides Protein Engineering
- **Authors:** Wang, Tang, Huang, ..., Yang
- **Venue:** Nature Machine Intelligence 2023
- **Link:** https://doi.org/10.1038/s42256-023-00691-9
- **Why study:** Uses MCTS + self-play for navigating protein fitness landscapes. The closest published work to ProteinDreamer in spirit — but uses a discriminative fitness oracle, not a full world model with latent state transitions. Key comparison paper.

### 6.3 Protein Sequence Design in Latent Space via Model-Based RL
- **Authors:** Lee, Vecchietti, Jung, Ro, Cha, Kim
- **Venue:** ICLR Workshop 2023
- **Link:** https://openreview.net/forum?id=OhjGzRE5N6o
- **Why study:** **The most directly related paper to ProteinDreamer.** Uses model-based RL in a latent protein space. Must study carefully to identify differentiation: ProteinDreamer adds (i) world-model framing (not just surrogate), (ii) active inference theory, (iii) structure-aware encoding, (iv) multi-step mutation planning with dreaming.

### 6.4 Model-Based RL for Protein Backbone Design
- **Authors:** Renard, Courtot, Reichlin, Bent
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2405.01983
- **Why study:** Applies AlphaZero-style MCTS to protein backbone design. Demonstrates model-based RL in protein structure space. Complements ProteinDreamer's focus on sequence-space mutations.

### 6.5 Accelerating Protein Engineering with Fitness Landscape Modelling and RL
- **Authors:** Sun, He, Deng, Liu, Zhao, Jiang, ...
- **Venue:** Nature Machine Intelligence 2025
- **Link:** https://doi.org/10.1038/s42256-025-01103-w
- **Why study:** Most recent paper combining fitness landscape models with RL (μFormer + μSearch). Uses a deep learning fitness predictor + RL-based search across mutation landscapes. Very relevant competitor — but no world model with latent dynamics or active inference. Key paper to differentiate from.

### 6.6 ProteinRL — RL with Generative Protein Language Models
- **Authors:** Sternke, Karpiak
- **Venue:** NeurIPS 2023 GenBio Workshop
- **Link:** https://openreview.net/forum?id=sWCsSKqkXa
- **Why study:** Fine-tunes a generative PLM via RL reward signals for property-directed design. Model-free RL approach — ProteinDreamer's model-based approach should be more sample-efficient. Good baseline.

---

## 7. Active Inference & Free Energy Principle

The theoretical backbone that distinguishes ProteinDreamer from generic RL-for-biology approaches.

### 7.1 Active Inference: The Free Energy Principle in Mind, Brain, and Behavior (Book)
- **Authors:** Parr, Pezzulo, Friston
- **Venue:** MIT Press 2022
- **Link:** https://doi.org/10.7551/mitpress/12441.001.0001
- **Why study:** The definitive textbook on active inference. Covers the mathematical framework (variational free energy, expected free energy, generative models, belief updating) that ProteinDreamer adapts to protein fitness landscapes. Essential for the theoretical contribution of the PhD.

### 7.2 Reinforcement Learning Through Active Inference
- **Authors:** Tschantz, Millidge, Seth, Buckley
- **Venue:** arXiv 2020
- **Link:** https://arxiv.org/abs/2002.12636
- **Why study:** Bridges active inference and standard RL. Shows how expected free energy minimisation subsumes reward maximisation + information gain. Provides the technical recipe for implementing active inference in a deep learning setting — directly applicable to ProteinDreamer.

### 7.3 Active Inference: Demystified and Compared
- **Authors:** Sajid, Ball, Parr, Friston
- **Venue:** Neural Computation 2021
- **Link:** https://doi.org/10.1162/neco_a_01357
- **Why study:** Accessible tutorial comparing active inference with RL, Bayesian RL, and control-as-inference. Clarifies the unique properties of active inference (intrinsic exploration, preference-based objectives) that motivate its use in ProteinDreamer.

### 7.4 The Free Energy Principle for Perception and Action: A Deep Learning Perspective
- **Authors:** Mazzaglia, Verbelen, Catal, Dhoedt
- **Venue:** Entropy 2022
- **Link:** https://doi.org/10.3390/e24020301
- **Why study:** Reviews deep learning implementations of active inference — VAE-based generative models, deep active inference agents. Provides the practical bridge between the theory (Friston) and the implementation (PyTorch).

### 7.5 Applications of the Free Energy Principle to Machine Learning and Neuroscience
- **Authors:** Millidge
- **Venue:** PhD Thesis / arXiv 2021
- **Link:** https://arxiv.org/abs/2107.00140
- **Why study:** Comprehensive treatment of connecting the Free Energy Principle with modern ML. Covers predictive coding networks, active inference agents, and variational methods. Good blueprint for how to frame a PhD thesis on this topic.

---

## 8. Fitness Landscapes & Protein Engineering (Experimental Data)

### 8.1 ProteinGym — Large-Scale Benchmarks for Protein Fitness Prediction and Design
- **Authors:** Notin, Kollasch, Ritter, ..., Marks
- **Venue:** NeurIPS 2023 Datasets & Benchmarks
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html
- **Why study:** **The primary benchmark** for ProteinDreamer. 217 substitution DMS assays covering diverse proteins. Standardised evaluation of zero-shot and supervised fitness prediction models. Must understand its structure, evaluation metrics, and model leaderboard.

### 8.2 Mega-Scale Stability Data — Tsuboyama et al.
- **Authors:** Tsuboyama, Dauparas, Chen, ..., Bhatt, Chothia, ..., Inoue, Marks, Hartl, Chiaromonte, Stein, ...
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06328-6
- **Why study:** Mega-scale experimental stability measurements (ΔΔG) for >500,000 protein variants across ~100 domains. The richest single training data source for ProteinDreamer's reward model and world model. Understanding this dataset is essential.

### 8.3 Machine Learning for Functional Protein Design (Review)
- **Authors:** Notin, Rollins, Gal, Sander, Marks
- **Venue:** Nature Biotechnology 2024
- **Link:** https://doi.org/10.1038/s41587-024-02127-0
- **Why study:** Comprehensive review of ML approaches to protein design — supervised, unsupervised, generative, and optimisation-based. Maps the entire field and identifies gaps. Essential for positioning BioDreamer/ProteinDreamer in the broader landscape.

### 8.4 Adaptive Machine Learning for Protein Engineering (Review)
- **Authors:** Hie, Yang
- **Venue:** Current Opinion in Structural Biology 2022
- **Link:** https://doi.org/10.1016/j.sbi.2021.11.002
- **Why study:** Reviews adaptive/active learning approaches to protein engineering. Discusses how ML models can guide iterative rounds of experiment and learning — exactly ProteinDreamer's setting, but without the world model/RL framing.

---

## 9. Equivariant & Geometric Deep Learning

### 9.1 GVP-GNN — Geometric Vector Perceptrons
- **Authors:** Jing, Eismann, Suriana, Townshend, Dror
- **Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2009.01411
- **Why study:** Equivariant graph neural network for protein structures using vector features. A candidate for ProteinDreamer's structure encoder — processes backbone coordinates while respecting rotation/translation symmetries.

### 9.2 EGNN — E(n) Equivariant Graph Neural Networks
- **Authors:** Satorras, Hoogeboom, Welling
- **Venue:** ICML 2021
- **Link:** https://arxiv.org/abs/2102.09844
- **Why study:** Simple, efficient equivariant GNN operating on point clouds. Core architecture candidate for MolWorld's encoder (maps atomic coordinates → latent state). Highly cited, well-understood.

### 9.3 PaiNN — Equivariant Message Passing for the Prediction of Tensorial Properties
- **Authors:** Schütt, Unke, Gastegger
- **Venue:** ICML 2021
- **Link:** https://arxiv.org/abs/2102.03150
- **Why study:** Equivariant message passing network for molecular properties. Used extensively in ML-for-chemistry; candidate encoder for MolWorld's molecular dynamics world model.

---

## 10. Neural ODEs & Continuous Dynamics Models

Relevant for both MolWorld (learned molecular dynamics) and CellDreamer (cellular dynamics).

### 10.1 Neural Ordinary Differential Equations
- **Authors:** Chen, Rubanova, Bettencourt, Duvenaud
- **Venue:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1806.07366
- **Why study:** Foundational paper introducing neural ODEs. Directly relevant to CellDreamer's dynamics model (latent ODE for cellular trajectories) and MolWorld's continuous-time dynamics. Must-read.

### 10.2 Latent SDEs — Score-Based Generative Modeling through SDEs
- **Authors:** Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole
- **Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2011.13456
- **Why study:** Connects score-based generative models with stochastic differential equations. The theoretical backbone for using diffusion/SDE models as dynamics models in MolWorld and CellDreamer.

---

## 11. Diffusion Models for Science

### 11.1 Score-Based Generative Modeling with SDEs (Song et al.)
- *(See 10.2 above)*

### 11.2 Diffusion Models for Molecules — A Comprehensive Survey
- **Authors:** Bilodeau, Jin, Bhatt, Bhagat, ...
- **Venue:** Various surveys 2023–2024
- **Why study:** Surveys the rapidly growing field of diffusion models for molecular generation, protein design, and drug discovery. Context for understanding where BioDreamer fits in the diffusion-for-biology ecosystem.

### 11.3 From Thermodynamics to Protein Design: Diffusion Models for Biomolecule Generation
- **Authors:** Li, Cadet, Medina-Ortiz, Davari, ...
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2501.02680
- **Why study:** Recent survey covering diffusion-based protein design from thermodynamic foundations. Useful for understanding the state-of-the-art and motivating the BioDreamer approach.

---

## Summary: Suggested Reading Order

For someone starting the BioDreamer project, a suggested reading sequence:

1. **World model foundations:** Ha & Schmidhuber (2018) → DreamerV3 → DIAMOND
2. **Protein design landscape:** Notin et al. review (2024) → ProteinMPNN → RFdiffusion → ESM-2/3
3. **RL for bio-sequences:** GFlowNets (Jain et al.) → EvoPlay → Lee et al. (latent-space MBRL)
4. **Active inference theory:** Parr et al. textbook → Tschantz et al. (RL through AI) → Sajid et al. (demystified)
5. **Data sources:** ProteinGym → Tsuboyama mega-scale → Replogle Perturb-seq
6. **Molecular dynamics ML:** MACE → NequIP → Timewarp
7. **Cellular models:** GEARS → CPA → scGPT
8. **Geometric DL:** GVP-GNN → EGNN → PaiNN
