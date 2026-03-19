# ProteinDreamer — PhD Proposal Literature Review

> **Deep literature for writing a PhD proposal on "ProteinDreamer: Model-Based RL for Protein Design".**
> Papers are organised by theme. Each entry explains *why* you must study it and *what* to extract from it for the proposal. Priority tags: 🔴 Must-read (cite in proposal), 🟡 Important context, 🟢 Good to know.

---

## Table of Contents

1. [World Models & Model-Based RL](#1-world-models--model-based-rl)
2. [RL for Protein / Biological Sequence Design — Direct Competitors](#2-rl-for-protein--biological-sequence-design--direct-competitors)
3. [Generative Protein Design — One-Shot Baselines](#3-generative-protein-design--one-shot-baselines)
4. [Protein Language Models & Fitness Prediction](#4-protein-language-models--fitness-prediction)
5. [Protein Structure Prediction — Cheap Oracles](#5-protein-structure-prediction--cheap-oracles)
6. [Deep Mutational Scanning & Fitness Landscape Data](#6-deep-mutational-scanning--fitness-landscape-data)
7. [Active Inference & Free Energy Principle — Theoretical Backbone](#7-active-inference--free-energy-principle--theoretical-backbone)
8. [Bayesian Optimisation & Active Learning for Protein Engineering](#8-bayesian-optimisation--active-learning-for-protein-engineering)
9. [Equivariant & Geometric Deep Learning for Proteins](#9-equivariant--geometric-deep-learning-for-proteins)
10. [GFlowNets & Alternative Generative Policies](#10-gflownets--alternative-generative-policies)
11. [Reviews & Surveys — Big-Picture Framing](#11-reviews--surveys--big-picture-framing)

---

## 1. World Models & Model-Based RL

These papers define the paradigm you are transferring to protein design. The proposal must demonstrate you deeply understand world models and can articulate *why* this paradigm is the right one for protein engineering.

---

### 🔴 1.1 DreamerV3 — Mastering Diverse Domains Through World Models
- **Authors:** Hafner, Pasukonis, Ba, Lillicrap
- **Venue:** arXiv 2023 / Nature 2025
- **Link:** https://arxiv.org/abs/2301.04104
- **What to extract:** The RSSM architecture (deterministic + stochastic latent states), symlog predictions, KL balancing, imagination-based actor-critic training. This is the **direct architectural template** for ProteinDreamer's world model. In the proposal, explain how you adapt each RSSM component to the protein domain: encoder → PLM/GVP, dynamics → mutation-conditioned transitions, reward head → fitness predictor, policy → mutation selector.
- **Key insight for proposal:** DreamerV3 generalises across 150+ environments with zero hyperparameter tuning — argue that this robustness makes it the right backbone for the heterogeneous protein fitness landscape (different proteins = different "environments").

### 🔴 1.2 DIAMOND — Diffusion for World Modeling
- **Authors:** Alonso, Jelley, Micheli, Fleuret
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.12399
- **What to extract:** Uses a diffusion model as the dynamics backbone instead of an RSSM. Produces more accurate long-horizon rollouts. Relevant because protein structure changes after mutation are naturally distributional (multiple plausible conformations). In the proposal, present diffusion-based dynamics as an alternative architecture for ProteinDreamer's world model.

### 🟡 1.3 IRIS — Transforming World Models with Discrete Tokens
- **Authors:** Micheli, Alonso, Fleuret
- **Venue:** ICML 2023
- **Link:** https://arxiv.org/abs/2209.00588
- **What to extract:** Discrete tokenisation + autoregressive transformers for world modelling. Protein sequences are inherently discrete (20 amino acid tokens), so IRIS's architecture is a natural fit. In the proposal, discuss this as an alternative world model backbone operating in sequence-token space rather than continuous latent space.

### 🟡 1.4 Ha & Schmidhuber — World Models (Original)
- **Authors:** Ha, Schmidhuber
- **Venue:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1803.10122
- **What to extract:** The conceptual framework: "learn an environment model, dream in it, train a controller on dreams." Cite this to establish the lineage and framing.

### 🔴 1.5 DynaPPO — Model-Based RL for Biological Sequence Design
- **Authors:** Angermueller, Dohan, Belanger, Colwell, Lucey, Skerry-Ryan, Steiner
- **Venue:** ICLR 2020
- **Link:** https://openreview.net/forum?id=HklxbgBKvr
- **What to extract:** **The foundational paper** for model-based RL in biological sequences. Uses a learned model of the sequence-fitness landscape as a surrogate, trains PPO on it. Critical to cite and differentiate from: DynaPPO uses a simple MLP surrogate, not a full world model with latent state transitions, structure awareness, or active inference exploration. In the proposal, position ProteinDreamer as the next-generation DynaPPO.

---

## 2. RL for Protein / Biological Sequence Design — Direct Competitors

These are the papers that reviewers / PIs will immediately compare you to. You must know them inside-out and clearly articulate what ProteinDreamer adds.

---

### 🔴 2.1 EvoPlay — Self-Play RL Guides Protein Engineering
- **Authors:** Wang, Tang, Huang, Pan, Yang, ...
- **Venue:** Nature Machine Intelligence 2023
- **Link:** https://doi.org/10.1038/s42256-023-00691-9
- **What to extract:** Uses MCTS + self-play for in-silico directed evolution. Trains a neural network to evaluate protein fitness and plans mutation trajectories via tree search. **Closest competitor in spirit.** Differentiation: EvoPlay uses a discriminative fitness oracle, not a generative world model with latent dynamics. It cannot "dream" new structures — only score proposed mutations. ProteinDreamer's world model predicts the *full state transition* (structure + fitness), enabling richer planning.

### 🔴 2.2 μFormer + μSearch — Accelerating Protein Engineering with Fitness Landscape Modelling and RL
- **Authors:** Sun, He, Deng, Liu, Zhao, Jiang, ...
- **Venue:** Nature Machine Intelligence 2025
- **Link:** https://doi.org/10.1038/s42256-025-01103-w
- **What to extract:** **Very recent and highly relevant.** Combines a deep learning fitness predictor (μFormer) with an RL-based search algorithm (μSearch) for multi-round simulated directed evolution. Achieves SOTA on multiple protein engineering benchmarks. Must differentiate: μSearch uses model-free RL with the fitness predictor as reward, not a full world model that predicts state transitions. No active inference, no structure-aware planning, no dreaming.

### 🔴 2.3 Protein Sequence Design in a Latent Space via Model-Based RL
- **Authors:** Lee, Vecchietti, Jung, Ro, Cha, Kim
- **Venue:** ICLR Workshop 2023
- **Link:** https://openreview.net/forum?id=OhjGzRE5N6o
- **What to extract:** **The most directly related paper.** Uses model-based RL in a VAE latent space to design protein sequences. Differentiation: (i) They use a simple VAE, not a pre-trained PLM + structure encoder. (ii) No world-model dynamics — the surrogate just predicts fitness, not how the protein evolves after a mutation. (iii) No active inference / exploration–exploitation balance. (iv) No multi-step mutation trajectory planning. This paper proves the concept works; ProteinDreamer is the full-scale version.

### 🔴 2.4 Model-Based RL for Protein Backbone Design
- **Authors:** Renard, Courtot, Reichlin, Bent
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2405.01983
- **What to extract:** Applies AlphaZero-style MCTS to design protein *backbone structures* (not sequences). Complementary to ProteinDreamer which operates in sequence space. Useful to cite as evidence that MCTS + world models work for proteins, but in a different design space.

### 🟡 2.5 ProteinRL — RL with Generative Protein Language Models for Property-Directed Sequence Design
- **Authors:** Sternke, Karpiak
- **Venue:** NeurIPS 2023 GenBio Workshop
- **Link:** https://openreview.net/forum?id=sWCsSKqkXa
- **What to extract:** Fine-tunes generative PLMs via RL reward signals. Model-free RL (no world model). Useful baseline comparison — argue that model-based approaches (ProteinDreamer) are more sample-efficient.

### 🟡 2.6 Reinforcement Learning for Sequence Design Leveraging Protein Language Models
- **Authors:** Subramanian, Sujit, Irtisam, Sain, ...
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2407.03154
- **What to extract:** Compares multiple RL algorithms (PPO, DQN, GFlowNets) for protein sequence design using PLM-based rewards. Provides an empirical study of RL algorithms in protein space — useful for justifying algorithm choices in ProteinDreamer.

### 🟡 2.7 Designing Biological Sequences via Meta-RL and Bayesian Optimisation
- **Authors:** Feng, Nouri, Muni, Bengio, ...
- **Venue:** arXiv 2022
- **Link:** https://arxiv.org/abs/2209.06259
- **What to extract:** Combines meta-RL with Bayesian optimisation for sequence design. Shows that learning to optimise across multiple fitness landscapes transfers to new ones. Relevant for ProteinDreamer's potential meta-learning extension.

---

## 3. Generative Protein Design — One-Shot Baselines

ProteinDreamer's pitch: these methods generate proteins in a single shot but **don't iterate, don't plan, don't explore.** You must know them to articulate the gap.

---

### 🔴 3.1 ProteinMPNN — Robust Deep Learning–Based Protein Sequence Design
- **Authors:** Dauparas, Anishchenko, ..., Baker
- **Venue:** Science 2022
- **Link:** https://doi.org/10.1126/science.add2187
- **What to extract:** The standard inverse-folding model. Given a backbone → design a sequence. One-shot, no iteration. Could serve as a sub-module within ProteinDreamer (re-score/re-design after each dreamed mutation step). Cite as the dominant paradigm that ProteinDreamer goes beyond.

### 🔴 3.2 RFdiffusion — De Novo Protein Design with Diffusion Models
- **Authors:** Watson, ..., Baker
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06415-8
- **What to extract:** Generates novel protein backbones via denoising diffusion on 3D coordinates. Designs binders, symmetric assemblies, motif scaffolds. One-shot — no iterative optimisation, no RL, no fitness-landscape navigation. Cite as proof that generative models work for proteins, but argue ProteinDreamer adds the missing iterative planning layer.

### 🔴 3.3 ESM-3 — Simulating 500M Years of Evolution with a Language Model
- **Authors:** Hayes et al. (EvolutionaryScale)
- **Venue:** bioRxiv 2024
- **Link:** https://doi.org/10.1101/2024.07.01.600583
- **What to extract:** Multi-modal protein foundation model (sequence + structure + function). Can generate functional de novo proteins. Represents the frontier of one-shot generative design. Argue that even ESM-3's powerful single-shot generation benefits from iterative refinement — ProteinDreamer provides exactly that.

### 🟡 3.4 EvoDiff — Protein Generation via Sequence Diffusion
- **Authors:** Alamdari, ..., Gitter
- **Venue:** Nature Biotechnology 2023
- **Link:** https://doi.org/10.1038/s41587-023-01902-9
- **What to extract:** Diffusion directly in discrete protein sequence space. Demonstrates sequence-space generative modelling — relevant architecture for ProteinDreamer's dynamics model.

### 🟡 3.5 Chroma — Generative Protein Design by Direct Structure Prediction
- **Authors:** Ingraham et al. (Generate Biomedicines)
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06728-8
- **What to extract:** Another diffusion approach with property conditioning. One-shot. Lists for completeness of the baseline landscape.

### 🟡 3.6 ProGen2 — Exploring the Protein Sequence Space with Language Models
- **Authors:** Nijkamp, Ruffolo, Weinstein, ..., Anandkumar
- **Venue:** Cell Systems 2023
- **Link:** https://doi.org/10.1016/j.cels.2023.10.002
- **What to extract:** Autoregressive protein language model for generation. Can be fine-tuned for family-specific generation. Potentially usable as the generative backbone in ProteinDreamer's policy.

---

## 4. Protein Language Models & Fitness Prediction

ProteinDreamer's world model needs encoders and fitness predictors. These papers define the building blocks.

---

### 🔴 4.1 ESM-2 — Language Models of Protein Evolution
- **Authors:** Lin, Akin, Rao, ..., Rives
- **Venue:** Science 2023
- **Link:** https://doi.org/10.1126/science.ade2574
- **What to extract:** **Primary sequence encoder for ProteinDreamer.** ESM-2 embeddings encode evolutionary and structural information. Zero-shot fitness prediction via masked marginals provides a strong baseline. Also introduces ESMFold (fast structure prediction). In the proposal, state that ESM-2 650M is your default sequence encoder.

### 🔴 4.2 ProteinGym — Large-Scale Benchmarks for Protein Fitness Prediction and Design
- **Authors:** Notin, Kollasch, Ritter, ..., Marks
- **Venue:** NeurIPS 2023 Datasets & Benchmarks
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html
- **What to extract:** **The primary evaluation benchmark.** 217 substitution DMS assays + 92 indel assays. Standardised leaderboard for fitness prediction models. You will benchmark ProteinDreamer's world model accuracy and design success against ProteinGym. Must understand: assay diversity, evaluation metrics (Spearman ρ, NDCG), supervised vs. zero-shot splits.

### 🔴 4.3 Multi-Scale Representation Learning for Protein Fitness Prediction
- **Authors:** Zhang, Notin, Huang, Lozano, ...
- **Venue:** NeurIPS 2024
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html
- **What to extract:** SOTA fitness prediction combining MSA-based and PLM-based features at multiple scales. If ProteinDreamer's world model should accurately predict fitness, understanding the best current predictors is essential. Their multi-scale approach may inspire the reward head architecture.

### 🟡 4.4 Fine-Tuning Protein Language Models with Deep Mutational Scanning
- **Authors:** Lafita, Gonzalez, Hossam, Smyth, ...
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2405.06729
- **What to extract:** Shows that fine-tuning ESM-2 on DMS data significantly improves variant effect prediction. Directly relevant to ProteinDreamer's world model training strategy — fine-tune PLM on fitness data from a specific assay before RL.

### 🟡 4.5 Tranception — Protein Fitness Prediction with Autoregressive Transformers
- **Authors:** Notin, Dias, Frazer, ..., Marks
- **Venue:** ICML 2022
- **Link:** https://proceedings.mlr.press/v162/notin22a.html
- **What to extract:** Autoregressive transformer for fitness prediction that uses retrieval-augmented inference with EVE. Strong zero-shot baseline. Could serve as the reward model backbone in ProteinDreamer.

---

## 5. Protein Structure Prediction — Cheap Oracles

ProteinDreamer uses structure predictors as cheap validation oracles (does the dreamed mutation fold correctly?) and as structure encoders.

---

### 🔴 5.1 AlphaFold2 — Highly Accurate Protein Structure Prediction
- **Authors:** Jumper, Evans, ..., Hassabis
- **Venue:** Nature 2021
- **Link:** https://doi.org/10.1038/s41586-021-03819-2
- **What to extract:** The breakthrough structure prediction model. In ProteinDreamer, AF2 (or its fast variants) serves as a "ground-truth oracle" for validating whether designed sequences fold to the intended structure. pLDDT scores serve as a proxy for designability.

### 🟡 5.2 ESMFold — Fast Single-Sequence Structure Prediction
- **Authors:** Lin et al. (part of ESM-2 paper)
- **Venue:** Science 2023
- **Link:** https://doi.org/10.1126/science.ade2574
- **What to extract:** 60x faster than AF2 with competitive accuracy. Enables rapid structure evaluation during ProteinDreamer's active learning loop. The speed makes it practical to evaluate thousands of dreamed candidates.

### 🟡 5.3 Protein Design Using Structure-Prediction Networks
- **Authors:** Wang, Watson, Lisanza
- **Venue:** Cold Spring Harbor Perspectives 2024
- **Link:** https://doi.org/10.1101/cshperspect.a041472
- **What to extract:** Review of using AF2/RoseTTAFold for protein design via hallucination and inverse folding. Provides context for how structure prediction has been repurposed for design — ProteinDreamer takes this further by adding RL-guided iterative planning.

---

## 6. Deep Mutational Scanning & Fitness Landscape Data

Understanding the training data is essential for a credible proposal. You need to show PIs that the data exists and you know how to use it.

---

### 🔴 6.1 Tsuboyama et al. — Mega-Scale Stability Measurements
- **Authors:** Tsuboyama, Dauparas, Chen, ..., Marks, Hartl, et al.
- **Venue:** Nature 2023
- **Link:** https://doi.org/10.1038/s41586-023-06328-6
- **What to extract:** >500,000 ΔΔG measurements across ~100 small protein domains. **The richest single training dataset** for ProteinDreamer's world model (stability prediction). Understand: assay methodology (protease sensitivity), coverage (single + double mutants), noise level, domain diversity.

### 🔴 6.2 Learning Protein Fitness Landscapes with DMS Data from Multiple Sources
- **Authors:** Chen, Zhang, Li, ..., Wang
- **Venue:** Cell Systems 2023
- **Link:** https://doi.org/10.1016/j.cels.2023.07.003
- **What to extract:** Addresses the challenge of learning fitness landscapes from heterogeneous DMS experiments. Discusses epistasis modelling and transfer learning across assays. Relevant because ProteinDreamer's world model must generalise across landscapes.

### 🟡 6.3 Machine Learning-Assisted Directed Evolution Navigates Combinatorial Epistatic Fitness Landscapes
- **Authors:** Wittmann, Yue, Arnold
- **Venue:** bioRxiv 2020 / ACS Synth. Biol. 2021
- **Link:** https://doi.org/10.1101/2020.12.04.408955
- **What to extract:** From Frances Arnold's group. Shows ML-guided directed evolution can navigate epistatic landscapes with minimal screening. Demonstrates the practical value of ProteinDreamer's multi-step planning — epistatic landscapes require sequential reasoning, not single-mutation greedy search.

### 🟡 6.4 Computational and Experimental Exploration of Protein Fitness Landscapes
- **Authors:** Sandhu, Chen, Matthews, Spence, ...
- **Venue:** Biochemistry 2025
- **Link:** https://doi.org/10.1021/acs.biochem.4c00673
- **What to extract:** Recent review of protein fitness landscape characteristics — ruggedness, epistasis, dimensionality. Good for the "Background" section of the proposal to explain why navigating fitness landscapes is hard and why world models help.

---

## 7. Active Inference & Free Energy Principle — Theoretical Backbone

This is ProteinDreamer's **unique theoretical contribution**: formalising protein design under the Free Energy Principle.

---

### 🔴 7.1 Active Inference: The Free Energy Principle in Mind, Brain, and Behavior (Textbook)
- **Authors:** Parr, Pezzulo, Friston
- **Venue:** MIT Press 2022
- **Link:** https://doi.org/10.7551/mitpress/12441.001.0001
- **What to extract:** The definitive reference. Chapters on: expected free energy (EFE) decomposition into pragmatic value (reward) + epistemic value (information gain); generative model specification; belief updating; policy selection via EFE minimisation. In the proposal, formally define: the generative model $p(o, s, \pi)$ in the protein context, the variational free energy $F$, and the expected free energy $G(\pi) = \mathbb{E}[\text{Risk}] - \mathbb{E}[\text{Information Gain}]$.

### 🔴 7.2 Reinforcement Learning Through Active Inference
- **Authors:** Tschantz, Millidge, Seth, Buckley
- **Venue:** arXiv 2020
- **Link:** https://arxiv.org/abs/2002.12636
- **What to extract:** **The technical bridge paper.** Shows how to implement active inference in a deep RL setting. Compares EFE with standard RL objectives. Demonstrates that active inference naturally balances exploration (uncertainty reduction) and exploitation (reward seeking). In the proposal, use this to justify why active inference is superior to standard RL for protein design: the fitness landscape is partially observed and expensive to query → exploration is crucial → EFE automatically provides it.

### 🔴 7.3 Active Inference: Demystified and Compared
- **Authors:** Sajid, Ball, Parr, Friston
- **Venue:** Neural Computation 2021
- **Link:** https://doi.org/10.1162/neco_a_01357
- **What to extract:** Clear, accessible comparison of active inference with RL, Bayesian RL, KL-control, and control-as-inference. Table 1 comparison is extremely useful for a proposal "Related Work" section. Extract the formal distinction: in RL the reward drives behaviour; in active inference the generative model + priors drive behaviour and reward is just one factor.

### 🟡 7.4 The Free Energy Principle for Perception and Action: A Deep Learning Perspective
- **Authors:** Mazzaglia, Verbelen, Catal, Dhoedt
- **Venue:** Entropy 2022
- **Link:** https://doi.org/10.3390/e24020301
- **What to extract:** Reviews deep learning implementations of active inference. Covers VAE-based generative models, deep active inference agents, and practical training recipes. Useful for the "Methods" section of the proposal — shows that implementing active inference with modern deep learning is feasible.

### 🟡 7.5 Applications of the Free Energy Principle to Machine Learning and Neuroscience
- **Authors:** Millidge
- **Venue:** PhD Thesis / arXiv 2021
- **Link:** https://arxiv.org/abs/2107.00140
- **What to extract:** A PhD thesis (!) that applies the FEP to ML. Good structural model for how to frame a thesis around this topic. Covers predictive coding, active inference, and connections to VAEs.

---

## 8. Bayesian Optimisation & Active Learning for Protein Engineering

These are the current approaches to iterative protein engineering. ProteinDreamer should be positioned as a more powerful alternative that uses world models instead of simple surrogates.

---

### 🔴 8.1 Adaptive Machine Learning for Protein Engineering
- **Authors:** Hie, Yang
- **Venue:** Current Opinion in Structural Biology 2022
- **Link:** https://doi.org/10.1016/j.sbi.2021.11.002
- **What to extract:** Reviews the ML-guided protein engineering loop: train surrogate → propose variants → test → retrain. Discusses acquisition functions (UCB, EI) for balancing exploration/exploitation. ProteinDreamer formalises this loop as MDP + world model + active inference. In the proposal, cite this to show that the community already does iterative design, but without the RL/world-model machinery.

### 🔴 8.2 Machine Learning for Protein Engineering (Book Chapter)
- **Authors:** Johnston, Fannjiang, Wittmann, Hie, Yang, Romero
- **Venue:** Springer 2023
- **Link:** https://doi.org/10.1007/978-3-031-37196-7_9
- **What to extract:** Comprehensive chapter covering supervised learning, Bayesian optimisation, RL, and active learning for protein engineering. Provides the unified framing that ProteinDreamer builds upon. Useful for the proposal's "Background" section.

### 🟡 8.3 Machine Learning to Navigate Fitness Landscapes for Protein Engineering
- **Authors:** Freschlin, Fahlberg, Romero
- **Venue:** Current Opinion in Biotechnology 2022
- **Link:** https://doi.org/10.1016/j.copbio.2022.102713
- **What to extract:** Review of ML models for fitness landscape navigation. Discusses data requirements, model types (linear, GP, neural nets), and experimental design. Context for why ProteinDreamer's world-model approach is needed: current surrogates are often simple and don't capture multi-step dynamics.

### 🟡 8.4 Machine Learning-Guided Co-Optimisation of Fitness and Diversity
- **Authors:** Ding, Chin, Zhao, Huang, Mai, ...
- **Venue:** Nature Communications 2024
- **Link:** https://doi.org/10.1038/s41467-024-50698-y
- **What to extract:** Active learning for enzyme engineering — co-optimises fitness and library diversity. Demonstrates the power of intelligent experimental design. ProteinDreamer's active inference naturally achieves this dual objective (EFE balances reward + information gain ≈ fitness + diversity).

### 🟡 8.5 Advancing Genetic Engineering with Active Learning
- **Authors:** Du, Wang, Jiang, Wang
- **Venue:** Briefings in Bioinformatics 2025
- **Link:** https://doi.org/10.1093/bib/bbaf286
- **What to extract:** Recent review of active learning strategies for genetic engineering. Covers uncertainty estimation methods (ensembles, Bayesian NNs, evidential DL). Useful for designing ProteinDreamer's uncertainty module that guides exploration.

---

## 9. Equivariant & Geometric Deep Learning for Proteins

Structure-aware encoding is key to ProteinDreamer's world model. These architectures process 3D protein structures while respecting physical symmetries.

---

### 🔴 9.1 GVP-GNN — Geometric Vector Perceptrons for Protein Structure
- **Authors:** Jing, Eismann, Suriana, Townshend, Dror
- **Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2009.01411
- **What to extract:** The leading equivariant GNN for protein structures. Uses scalar and vector features on backbone atoms. **Primary candidate for ProteinDreamer's structure encoder.** In the proposal, state: "We use GVP-GNN to encode predicted structures into geometric features that augment ESM-2 sequence embeddings."

### 🟡 9.2 EGNN — E(n) Equivariant Graph Neural Networks
- **Authors:** Satorras, Hoogeboom, Welling
- **Venue:** ICML 2021
- **Link:** https://arxiv.org/abs/2102.09844
- **What to extract:** Simple, efficient equivariant GNN. Alternative to GVP-GNN with fewer parameters. Useful if you want a lighter encoder.

---

## 10. GFlowNets & Alternative Generative Policies

GFlowNets are a competing paradigm for diverse biological sequence generation. You should know them to compare and potentially integrate.

---

### 🔴 10.1 Biological Sequence Design with GFlowNets
- **Authors:** Jain, Bengio, Hernandez-Garcia, ..., Bengio
- **Venue:** ICML 2022
- **Link:** https://proceedings.mlr.press/v162/jain22a.html
- **What to extract:** Introduces GFlowNets for sequence design — a diversity-seeking alternative to RL that samples proportionally to reward. Compares with DynaPPO. In the proposal, acknowledge GFlowNets as an alternative policy class and argue that ProteinDreamer's world-model framework is complementary: the world model provides the environment, and GFlowNets or standard RL can serve as the policy.

### 🟡 10.2 GFlowNet-Assisted Biological Sequence Editing
- **Authors:** Ghari, Tseng, Eraslan, Lopez, ...
- **Venue:** NeurIPS 2024
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/c14760740573001c0d18d58879a6a305-Abstract-Conference.html
- **What to extract:** Uses GFlowNets for sequence *editing* (not de novo design) — closer to ProteinDreamer's mutation-based approach. The GFlowNet proposes edits; a reward model scores them. Could be used as ProteinDreamer's policy module.

### 🟡 10.3 Improved Off-Policy RL in Biological Sequence Design
- **Authors:** Kim, Kim, Yun, Choi, Bengio, ...
- **Venue:** arXiv 2024
- **Link:** https://arxiv.org/abs/2410.04461
- **What to extract:** Improves GFlowNet training by mixing offline and on-policy data. Relevant for ProteinDreamer's training efficiency — the world model enables offline data collection (dreaming) that augments real experimental data.

---

## 11. Reviews & Surveys — Big-Picture Framing

Use these to write the proposal's Introduction and Background with authority.

---

### 🔴 11.1 Machine Learning for Functional Protein Design
- **Authors:** Notin, Rollins, Gal, Sander, Marks
- **Venue:** Nature Biotechnology 2024
- **Link:** https://doi.org/10.1038/s41587-024-02127-0
- **What to extract:** Comprehensive state-of-the-field review. Maps all ML approaches to protein design: supervised, unsupervised, generative, optimisation. **Use this to position ProteinDreamer in the big picture.** The review identifies iterative optimisation as an under-explored direction — ProteinDreamer fills that gap.

### 🔴 11.2 A Model-Centric Review of Deep Learning for Protein Design
- **Authors:** Kyro, Qiu, Batista
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2502.19173
- **What to extract:** Very recent review organising protein design by model architecture (autoregressive, diffusion, flow-matching, etc.). Useful to show awareness of the latest landscape and to identify where world-model-based approaches (ProteinDreamer) are missing from the taxonomy.

### 🟡 11.3 From Thermodynamics to Protein Design: Diffusion Models for Biomolecule Generation
- **Authors:** Li, Cadet, Medina-Ortiz, Davari, ...
- **Venue:** arXiv 2025
- **Link:** https://arxiv.org/abs/2501.02680
- **What to extract:** Covers diffusion-based protein design with a thermodynamic perspective. Relevant for understanding the physical grounding of generative models, which ProteinDreamer's world model should respect.

### 🟡 11.4 From Predicting to Decision Making: RL in Biomedicine
- **Authors:** Liu, Zhang, Hou, Yang, ...
- **Venue:** WIREs Computational Molecular Science 2024
- **Link:** https://doi.org/10.1002/wcms.1723
- **What to extract:** Broad review of RL in biomedicine including drug design, molecular optimisation, and protein engineering. Useful for the "Related Work" section to show the broader RL-for-biology landscape.

---

## Summary: Reading Priority for Proposal Writing

### Phase 1 — Core (Read first, form the backbone of the proposal)
1. DreamerV3 (Hafner et al., 2023)
2. DynaPPO (Angermueller et al., 2019)
3. EvoPlay (Wang et al., 2023)
4. μFormer+μSearch (Sun et al., 2025)
5. Lee et al. — MBRL for protein in latent space (2023)
6. ESM-2 (Lin et al., 2023)
7. ProteinGym (Notin et al., 2023)
8. Tsuboyama mega-scale stability (2023)
9. Parr et al. — Active Inference textbook (2022)
10. Tschantz et al. — RL through Active Inference (2020)

### Phase 2 — Baselines & Positioning
11. ProteinMPNN (Dauparas et al., 2022)
12. RFdiffusion (Watson et al., 2023)
13. ESM-3 (Hayes et al., 2024)
14. GFlowNets for bio-sequences (Jain et al., 2022)
15. Notin et al. — ML for Functional Protein Design review (2024)
16. Hie & Yang — Adaptive ML for Protein Engineering (2022)

### Phase 3 — Technical Architecture
17. DIAMOND (Alonso et al., 2024)
18. GVP-GNN (Jing et al., 2021)
19. AlphaFold2 (Jumper et al., 2021)
20. Sajid et al. — Active Inference Demystified (2021)
21. Multi-scale fitness prediction (Zhang et al., 2024)

### Phase 4 — Extended Context
22. Remaining papers from Sections 6, 8, 9, 10
