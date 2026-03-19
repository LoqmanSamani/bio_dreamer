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

**Approach:** Frame protein design as a Markov Decision Process:
- **State** = sequence $\mathbf{x} \in \{A,\ldots,Y\}^L$ + predicted structure + property estimates.
- **Action** = mutation (substitution, insertion, deletion, loop redesign).
- **Transition model (world model)** = predicts how structure and fitness change after a mutation, learned from deep mutational scanning data + structure predictors.
- **Reward** = target fitness (stability $\Delta\Delta G$, affinity $K_d$, activity $k_{cat}$, or multi-objective).
- **Policy** = RL agent (SAC/PPO in latent space or MCTS over mutation trees) that plans mutation paths in the world model before evaluation.

**Theoretical backbone — Active Inference:**
The system is grounded in the *Free Energy Principle* (Friston, 2010): the agent maintains a generative model of the protein fitness landscape and selects mutations that minimise *expected free energy* — a quantity that naturally trades off exploitation (seeking high fitness) and exploration (reducing model uncertainty). This provides:
- A principled exploration–exploitation balance in the vast, rugged sequence space.
- A Bayesian framework for active learning: after each batch of real experiments, the world model is updated, uncertainty decreases, and the policy refines.
- A novel theoretical bridge connecting computational neuroscience, Bayesian inference, RL, and protein biology.

**Data:** ProteinGym (200+ DMS assays), Tsuboyama mega-scale stability data, AlphaFold2/ESMFold as cheap structure oracles, BRENDA/EnzML for enzyme activity.

**Deliverable:** `protein-dreamer` — a framework for model-based RL protein engineering with active learning. User provides wild-type + objective → system dreams optimal mutation paths → ranks candidates → optionally interfaces with wet-lab validation loop.

**Why this is the strongest PhD candidate:**
1. **Clear novelty:** No existing work combines world models + RL + active inference for iterative protein design. One-shot methods dominate; this is the first principled sequential approach.
2. **Rich theoretical contribution:** The Active Inference framing is publishable on its own and connects to a vibrant neuroscience/AI theory community.
3. **Practical impact:** Directly applicable to therapeutic antibody engineering, enzyme design, vaccine development — attractive to both academic and industry labs.
4. **Data availability:** ProteinGym and mega-scale datasets make training feasible without requiring your own wet lab.
5. **Scalable scope:** Can start with a simple sequence-only world model and scale up to structure-aware, multi-objective, active-learning variants — natural PhD progression.

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

Although each pillar targets a different biological scale, they share a common computational skeleton:

```
┌─────────────────────────────────────────────────────┐
│                   BioDreamer Core                   │
│                                                     │
│  ┌───────────┐   ┌──────────────┐   ┌───────────┐   │
│  │  Encoder  │──▶│ Latent World │──▶│  Decoder  │   │
│  │ (domain-  │   │   Model      │   │ (domain-  │   │
│  │  specific)│   │ (shared arch)│   │  specific)│   │
│  └───────────┘   └──────┬───────┘   └───────────┘   │
│                         │                           │
│                  ┌──────▼───────┐                   │
│                  │ Reward Model │                   │
│                  └──────┬───────┘                   │
│                         │                           │
│                  ┌──────▼───────┐                   │
│                  │    Policy    │                   │
│                  │  (RL Agent)  │                   │
│                  └──────────────┘                   │
└─────────────────────────────────────────────────────┘

Instantiated as:
  • MolWorld:        GNN encoder → Latent diffusion dynamics → Coordinate decoder
  • ProteinDreamer:  PLM+GVP encoder → Conditional transformer/diffusion → Fitness decoder
  • CellDreamer:     scRNA VAE encoder → Neural ODE/SDE → Gene expression decoder
```

This shared structure means:
- **Code reuse:** A single `biodreamer` library with pluggable encoders, dynamics models, decoders, and reward heads.
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

**Year 1 — Foundations**
- Literature review: world models (DreamerV3, IRIS), protein design (ProteinMPNN, RFdiffusion, ESM-3), fitness prediction, Active Inference.
- Build the basic pipeline: sequence encoder (ESM-2 embeddings) → simple world model (MLP / small transformer predicting DMS fitness) → policy (PPO on mutation actions).
- Benchmark on ProteinGym single-mutant landscapes.
- **Target output:** Workshop paper or preprint on "world models for protein fitness landscapes."

**Year 2 — Scaling and Structure**
- Integrate structure-aware encoding (GVP-GNN, predicted structures from ESMFold).
- Upgrade world model to latent diffusion or autoregressive transformer.
- Implement Active Inference formulation — compare with standard RL baselines.
- Multi-step mutation planning: benchmark on known evolutionary paths (e.g., TEM-1 β-lactamase evolution).
- **Target output:** Top-venue paper (NeurIPS / ICML / Nature Methods).

**Year 3 — Multi-objective, Active Learning, and Validation**
- Multi-objective optimisation (stability + activity + expressibility).
- Active learning loop: world model proposes candidates → cheap oracle (ESMFold / ProteinMPNN inverse folding) validates → model updates.
- If wet-lab collaboration available: real experimental validation on a model enzyme or antibody.
- Optionally extend to MolWorld for MD-based validation of top candidates.
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
