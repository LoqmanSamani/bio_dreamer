# BioDreamer — World Models for Biological Design

> *Teaching machines to dream about biology so we don't have to wait for every experiment.*

---

## 1. Vision

Biology operates at every scale — from atoms jiggling in a protein's binding pocket, to amino-acid mutations reshaping fitness landscapes, to thousands of genes orchestrating cell fate. At each scale, the core challenge is the same: **the real environment is prohibitively expensive to query** (MD simulations, wet-lab assays, CRISPR screens), yet intelligent design demands exploring vast combinatorial spaces.

**BioDreamer** is a unified research programme that applies the *world model* paradigm from model-based reinforcement learning to biological systems across three scales:

| Scale | Module | Environment | Agent Action | Reward |
|---|---|---|---|---|
| **Atomic** | MolWorld | Molecular dynamics of biomolecules | Force/parameter changes, mutations | Binding ΔG, stability |
| **Protein** | ProteinDreamer | Protein fitness landscape | Sequence mutations / edits | Stability, function, affinity |
| **Cellular** | CellDreamer | Gene regulatory network dynamics | Gene knockouts, drug perturbations | Target cell state / phenotype |

The shared thesis: **learn a latent-space simulator (world model) of each biological environment, then use model-based RL to plan optimal interventions "in imagination" — replacing brute-force simulation or experimentation with intelligent, amortised, in-silico reasoning.**

---

## 2. The Three Pillars

### 2.1 MolWorld — World Models for Molecular Dynamics

**Problem:** Classical MD simulations solve Newton's equations step-by-step with empirical force fields or ab initio potentials. Simulating biologically relevant timescales (microseconds–milliseconds) for even a single protein requires days–weeks on GPU clusters. ML surrogates (MACE, NequIP, TorchMD-NET) accelerate the force evaluation but still integrate trajectories one timestep at a time.

**Approach:** Train a full *world model* in a learned latent space:
- **Encoder** (SE(3)-equivariant GNN) compresses atomic coordinates → latent state $z_t$.
- **Dynamics model** (latent diffusion / neural SDE) evolves $z_t \to z_{t+1}$, potentially with variable step sizes and coarse-grained time jumps.
- **Decoder** reconstructs coordinates, contact maps, and observables.
- **Reward model** predicts properties of interest ($\Delta G$, $T_m$, SASA).
- **Policy** (actor-critic in latent space) proposes mutations or simulation parameters to optimise reward.

**Data:** ATLAS protein dynamics dataset, DE Shaw long MD trajectories, PDBBind, Martini 3 coarse-grained datasets.

**Deliverable:** `molworld` — an open-source framework for training latent world models on MD data and doing model-based RL for molecular optimisation.

---

### 2.2 ProteinDreamer — Model-Based RL for Protein Design ⭐ (Core)

**Problem:** Protein engineering today is either *one-shot generative* (ProteinMPNN, RFdiffusion, ESM-3 — generate a sequence, hope it works) or *directed evolution* (random mutagenesis + screening — expensive, slow, combinatorially limited). Neither approach does what a skilled engineer would: **plan a multi-step mutation strategy by mentally simulating outcomes before committing**.

**Approach:** Frame protein design as a Markov Decision Process with a **JEPA-based world model** that operates entirely in latent space:
- **State** = sequence $\mathbf{x} \in \{A,\ldots,Y\}^L$ + predicted structure + property estimates → encoded into latent state $z_t$.
- **Action** = mutation (substitution, insertion, deletion, loop redesign) — framed as a **causal intervention** on $z_t$.
- **Transition model (JEPA world model)** = predicts the post-mutation latent state $\hat{z}_{t+1}$ given $(z_t, a_t)$. All planning happens in latent space — **no decoder needed during imagination rollouts**.
- **Reward** = target fitness (stability $\Delta\Delta G$, affinity $K_d$, activity $k_{cat}$, or multi-objective) — predicted from $z_t$ by a reward head.
- **Policy** = RL agent (SAC/PPO in latent space or MCTS over mutation trees) that plans mutation paths in the world model before evaluation.

**World model backends (the library supports multiple architectures):**

| Backend | Type | Predictor | Uncertainty | Speed | Distributional |
|---|---|---|---|---|---|
| **Latent Diffusion JEPA** | Generative | Conditional diffusion in latent space ($d \approx 256$–$512$) | Built-in (sample variance) | Medium | ✅ Multi-modal |
| **Energy-Based JEPA** | Non-generative | Deterministic Transformer/MLP + SIGReg | External (ensembles / evidential DL) | Fast | ❌ Single-mode |
| **RSSM (DreamerV3)** | Reconstruction-based | Stochastic + deterministic recurrent | KL-based | Medium | Partial (Gaussian) |
| **Discrete Tokens (IRIS)** | Autoregressive | Transformer over VQ tokens | Predictive entropy | Medium | Partial |

The primary architectures for the PhD are **Latent Diffusion JEPA** (Architecture A) and **Energy-Based JEPA** (Architecture B), with RSSM and discrete-token backends available as baselines.

**Architecture A — Latent Diffusion JEPA (Primary):**
The JEPA encoder maps protein states (ESM-2 sequence embeddings + GVP-GNN structure features) to latent $z_t$. The predictor is a **conditional denoising diffusion model** that generates samples from $p_\theta(z_{t+1} | z_t, a_t)$ via iterative denoising in the compact latent space. This captures the multi-modal, stochastic nature of protein fitness landscapes — a single mutation can lead to distinct structural/functional outcomes. The diffusion sample variance provides built-in uncertainty estimates that feed directly into the Active Inference exploration term (Expected Free Energy). Training loss: JEPA energy + denoising score matching.

**Architecture B — Energy-Based JEPA (Alternative):**
Same JEPA encoder, but the predictor is a deterministic Transformer/MLP mapping $(z_t, a_t) \to \hat{z}_{t+1}$, trained with prediction loss + SIGReg (Sketched-Isotropic-Gaussian Regularizer; LeWorldModel, Maes et al., 2026) on the encoder output $z_t$. SIGReg enforces $z_t \sim \mathcal{N}(0, I)$ via random projections + Epps-Pulley normality testing — only 1 hyperparameter ($\lambda$) vs. VICReg's 6-7, much simpler to tune. No decoder. Single forward pass → faster inference, but averages over modes. Suitable for rapid screening or when combined with an external uncertainty module.

**Theoretical backbone — Active Inference:**
The system is grounded in the *Free Energy Principle* (Friston, 2010). The deep connection between JEPA and Active Inference: both minimise variational free energy. The JEPA energy $E_\theta(z_t, a_t, z_{t+1})$ scores prediction consistency — mathematically analogous to the prediction error in $F = D_{KL}[q(s) \| p(s)] - \mathbb{E}_q[\ln p(o|s)]$. The agent selects mutations that minimise *expected free energy* $G(\pi)$, which naturally decomposes into:
- **Pragmatic value** (exploitation): seek high fitness states.
- **Epistemic value** (exploration): seek states where the world model is uncertain (reduce model uncertainty).
For Architecture A, the diffusion predictor provides a rich distributional estimate that improves both terms. For Architecture B, external uncertainty modules (ensembles, evidential DL) approximate the epistemic term.

**Data:** ProteinGym (200+ DMS assays), Tsuboyama mega-scale stability data, AlphaFold2/ESMFold as cheap structure oracles, BRENDA/EnzML for enzyme activity.

**Deliverable:** `protein-dreamer` — a modular framework for model-based RL protein engineering. User provides wild-type + objective → selects world model backend → system dreams optimal mutation paths → ranks candidates → optionally interfaces with wet-lab validation loop. The library design supports swapping any component (encoder, predictor, reward head, policy) independently.

**Why this is the strongest PhD candidate:**
1. **Clear novelty:** No existing work combines JEPA world models + latent diffusion dynamics + active inference for iterative protein design. The latent diffusion JEPA architecture (Architecture A) is entirely new.
2. **Rich theoretical contribution:** The JEPA–Active Inference unification is publishable on its own and connects to vibrant neuroscience/AI theory communities.
3. **Practical impact:** Directly applicable to therapeutic antibody engineering, enzyme design, vaccine development — attractive to both academic and industry labs.
4. **Data availability:** ProteinGym and mega-scale datasets make training feasible without requiring your own wet lab.
5. **Scalable scope:** Can start with Energy-Based JEPA (simpler) and scale to Latent Diffusion JEPA, then multi-objective, active-learning variants — natural PhD progression.
6. **Library design:** The multi-backend approach ensures the PhD produces a reusable tool, not just a paper.

---

### 2.3 CellDreamer — World Models for Gene Regulatory Network Dynamics

**Problem:** Understanding and controlling cellular behaviour (differentiation, reprogramming, drug response) requires modelling gene regulatory dynamics. Classical approaches (Boolean networks, ODE systems) don't scale to genome-wide networks. Recent ML models (GEARS, CPA, scGPT) predict perturbation outcomes but only at steady state — they don't model temporal trajectories or enable multi-step intervention planning.

**Approach:** Learn a world model of cellular dynamics from perturbation data:
- **Encoder** (VAE on scRNA-seq profiles) maps transcriptomic snapshot → latent cell state $z_t$.
- **Dynamics model** (Neural ODE/SDE or conditional diffusion) evolves $z_t \to z_{t+\Delta t}$, conditioned on perturbation actions.
- **Actions** = gene knockouts (CRISPRi/a), drug treatments (compound + dose + timing), or combinations.
- **Reward** = distance to target cell state (e.g., apoptosis, cardiomyocyte fate, high metabolite production).
- **Policy** = RL agent that plans multi-step, multi-gene perturbation sequences.

**Data:** Replogle et al. (2022) genome-wide Perturb-seq, Norman et al. (2019) combinatorial CRISPRa, sci-Plex drug perturbations, developmental spatial transcriptomics (Stereo-seq, MERFISH).

**Deliverable:** `cell-dreamer` — a framework for learning cell-dynamics world models and planning perturbation strategies with RL.

---

## 3. Unified Architecture

Although each pillar targets a different biological scale, they share a common computational skeleton based on the **JEPA (Joint-Embedding Predictive Architecture)** paradigm: encode observations into latent states, predict transitions in latent space, and train policies on imagined trajectories — all without requiring a decoder during planning.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BioDreamer Core                              │
│                                                                      │
│  ┌───────────────┐   ┌─────────────────────┐   ┌───────────────┐    │
│  │ Context       │   │   JEPA Predictor     │   │ Target        │    │
│  │ Encoder f_θ   │──▶│ (World Model         │   │ Encoder f_θ̄   │    │
│  │ (domain-      │   │  Dynamics)           │   │ (EMA of f_θ)  │    │
│  │  specific)    │   │                      │   │               │    │
│  └───────────────┘   │ Options:             │   └───────┬───────┘    │
│         │            │ • Latent Diffusion   │           │            │
│         │            │ • Deterministic MLP  │           │            │
│         │            │ • RSSM (legacy)      │           │            │
│         ▼            │ • Discrete Tokens    │           │            │
│     z_t, a_t  ──────▶│                      │──▶ ẑ_{t+1}│            │
│                      └──────────┬───────────┘     vs    │            │
│                                 │               z̄_{t+1} ◄────────── │
│                          ┌──────▼───────┐      (JEPA energy)        │
│                          │ Reward Head  │                            │
│                          │ (Fitness)    │                            │
│                          └──────┬───────┘                            │
│                                 │                                    │
│                          ┌──────▼───────┐                            │
│                          │   Policy     │                            │
│                          │  (RL Agent)  │                            │
│                          └──────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘

Instantiated as:
  • MolWorld:        SE(3)-GNN encoder → Latent diffusion/SDE dynamics → Coordinate decoder (needed for MD)
  • ProteinDreamer:  ESM-2+GVP encoder → Latent Diffusion JEPA or Energy-Based JEPA → Fitness head (no decoder during planning)
  • CellDreamer:     scRNA VAE encoder → Neural ODE/SDE or Latent Diffusion → Gene expression decoder
```

This shared structure means:
- **Code reuse:** A single `biodreamer` library with pluggable encoders, dynamics models (JEPA predictors), reward heads, and policies. Each component can be swapped independently.
- **Architecture flexibility:** Users choose between generative (latent diffusion) and non-generative (energy-based JEPA) world model backends depending on their speed–accuracy tradeoff.
- **Transfer learning:** Insights from MolWorld (atomic scale) can inform the structure-prediction component of ProteinDreamer; CellDreamer perturbation predictions can serve as downstream validation for protein designs.
- **Unified publications:** Each pillar is a paper; the framework itself is a systems/software paper.

---

## 4. Idea Rankings for PhD Proposal Value

Below, each idea (and key combinations) is ranked by its suitability as the centrepiece of a PhD proposal — considering novelty, feasibility, theoretical depth, data availability, publication potential, and attractiveness to PIs.

### Tier 1 — Strongest PhD Proposals

| Rank | Idea | Score | Rationale |
|------|------|-------|-----------|
| **1** | **ProteinDreamer (standalone)** | ★★★★★ | Clearest novelty gap (no world-model RL for protein design exists), rich theoretical angle (Active Inference), abundant training data (ProteinGym, mega-scale DMS), huge practical impact (enzyme/antibody engineering), and a natural 3–4 year progression from simple to complex. Most PIs in computational biology or ML-for-science will immediately see the value. |
| **2** | **ProteinDreamer + MolWorld (combined)** | ★★★★★ | ProteinDreamer as the core, with MolWorld providing a learned MD simulator as the "ground-truth oracle" for the world model's structure/stability predictions. This is a compelling multi-scale story: the agent dreams mutations (ProteinDreamer) and validates critical candidates in a learned MD environment (MolWorld) — all in silico. Adds depth and a second line of publications, but increases scope. Best if the PhD is 4+ years or has strong computational resources. |

### Tier 2 — Strong PhD Proposals

| Rank | Idea | Score | Rationale |
|------|------|-------|-----------|
| **3** | **ProteinDreamer + CellDreamer (combined)** | ★★★★☆ | ProteinDreamer designs optimised proteins; CellDreamer validates their downstream cellular effect (e.g., does this engineered transcription factor actually drive the target gene programme?). Multi-scale story from protein to cell. Strong, but the cell-level component requires more data curation and the connection is less tight than ProteinDreamer + MolWorld. |
| **4** | **MolWorld (standalone)** | ★★★★☆ | Solid novelty (world models for MD is unexplored), clear benchmarks, and connects to the large ML-for-MD community. Slightly lower than ProteinDreamer because the application story (what do you *do* with the learned simulator?) is less crisp — optimising MD parameters is less compelling to bio-oriented PIs than designing proteins. Strongest for PIs in ML-for-physical-sciences or computational chemistry. |

### Tier 3 — Good but Narrower

| Rank | Idea | Score | Rationale |
|------|------|-------|-----------|
| **5** | **CellDreamer (standalone)** | ★★★☆☆ | Interesting and timely (Perturb-seq data is exploding), but competes with a crowded field (GEARS, CPA, scGPT, scFoundation models). The world-model + RL framing adds novelty, but the dynamics data (time-resolved perturbation scRNA-seq) is still scarce, making training challenging. Best suited as a later-stage extension or if the student has strong genomics/single-cell expertise. |
| **6** | **Full BioDreamer (all three combined)** | ★★★☆☆ | Maximally ambitious — a unified framework across atomic, protein, and cellular scales. Intellectually beautiful, but likely **too broad for a single PhD**. High risk of spreading too thin. Better framed as a long-term lab vision with the PhD focusing on one pillar (ProteinDreamer) and sketching the others as future work. |

---

## 5. Recommended PhD Strategy

### Primary focus: **ProteinDreamer**

This is the core of the PhD — a complete, novel contribution that is feasible within 3–4 years.

### Suggested PhD timeline (approximate):

**Year 1 — Foundations & Energy-Based JEPA**
- Literature review: JEPA world models (Causal-JEPA, LeWorldModel), DreamerV3 (historical), protein design (ProteinMPNN, RFdiffusion, ESM-3), fitness prediction, Active Inference.
- Build Architecture B: ESM-2 + GVP-GNN encoder (both output $\mathbb{R}^{L \times 1280}$) → FusionMLP → $z_t$ with SIGReg → Energy-Based JEPA predictor (deterministic, prediction loss) → fitness reward head → PPO policy in latent space.
- Benchmark on ProteinGym single-mutant landscapes.
- **Target output:** Workshop paper or preprint on "JEPA World Models for Protein Fitness Landscapes."

**Year 2 — Latent Diffusion JEPA & Structure-Awareness**
- Build Architecture A: Replace deterministic predictor with conditional latent diffusion model. Train with combined JEPA energy + denoising score-matching loss.
- Integrate structure-aware encoding (GVP-GNN, predicted structures from ESMFold).
- Implement Active Inference formulation — compare EFE-based exploration with standard RL baselines. Leverage diffusion sample variance for epistemic uncertainty.
- Multi-step mutation planning: benchmark on known evolutionary paths (e.g., TEM-1 β-lactamase evolution).
- **Target output:** Top-venue paper (NeurIPS / ICML / Nature Methods) — "Latent Diffusion JEPA for Model-Based Protein Design."

**Year 3 — Multi-objective, Active Learning, and Validation**
- Multi-objective optimisation (stability + activity + expressibility).
- Active learning loop: diffusion world model proposes candidates → cheap oracle (ESMFold / ProteinMPNN) validates → model updates → uncertainty-driven exploration refines.
- Comparative study: Architecture A vs. B vs. DreamerV3 RSSM vs. IRIS discrete tokens — systematic ablation.
- If wet-lab collaboration available: real experimental validation on a model enzyme or antibody.
- **Target output:** Journal paper (Nature Computational Science / PNAS) + framework release.

**Year 4 (if applicable) — Extensions and Thesis**
- Extend to CellDreamer or MolWorld as a second pillar.
- Write and defend thesis: "World Models for Biological Design" with ProteinDreamer as the core and extensions as future directions.
- **Target output:** Thesis + 1 additional paper on the extension.

---

## 6. Key Selling Points for PI Outreach

When reaching out to PIs, emphasise:

1. **Novel framing:** "I want to bring world models — the most successful paradigm in model-based RL — to protein engineering. No one has done this."
2. **Theoretical depth:** "I ground the approach in Active Inference / Free Energy Principle, providing a principled exploration–exploitation framework for navigating fitness landscapes."
3. **Practical relevance:** "This has direct applications to therapeutic antibody design, enzyme engineering, and vaccine development."
4. **Feasibility:** "Training data exists (ProteinGym, mega-scale DMS). I don't need a wet lab to make progress — but wet-lab validation would make the work even stronger."
5. **Publication potential:** "The project naturally decomposes into 3–4 papers across ML venues (NeurIPS, ICML) and biology venues (Nature Methods, PNAS)."
6. **Extensibility:** "ProteinDreamer is the core, but the world-model framework generalises to molecular dynamics (MolWorld) and cellular dynamics (CellDreamer) — a long-term research programme."

---

## 7. Next Steps

- [ ] Finalise which idea / combination to lead the proposal with.
- [ ] Deep literature review for ProteinDreamer — identify exact novelty gap and related work.
- [ ] Design the project directory structure and codebase architecture.
- [ ] Write a 2-page PhD proposal draft.
- [ ] Identify target PIs and labs (computational biology + ML intersection).
- [ ] Build a minimal proof-of-concept: world model on a single DMS dataset.
