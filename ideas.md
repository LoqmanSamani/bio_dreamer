### MolWorld — World Models for Molecular Dynamics

- **The big idea:** World models (like DreamerV3, IRIS, DIAMOND) learn to simulate environments internally so an agent can "imagine" trajectories without ever stepping into the real environment. Apply this paradigm to molecular systems: train a world model that predicts how a protein–ligand complex or a solvated biomolecule evolves over time — essentially a learned, GPU-native molecular dynamics (MD) simulator operating in a compact latent space.

- **Why it's novel:** Classical MD simulations are computationally brutal — simulating microseconds of protein dynamics can take days on GPU clusters. ML surrogates exist (neural force fields like MACE, NequIP, TorchMD-NET), but they replace only the potential energy surface; they still integrate Newton's equations step-by-step. **Nobody has recast the full MD pipeline as a world model** with (i) a learned latent state representation, (ii) a latent transition/dynamics model, (iii) a reward/property predictor, and (iv) a policy that plans and acts within the learned simulator. This latent-space formulation allows variable-length jumps, coarse-grained reasoning, and amortised long-horizon rollouts that classical integrators cannot offer.

- **What it combines:** World models (model-based RL) + structural biology + equivariant geometric deep learning + diffusion/score-based generative models (as the stochastic dynamics backbone).

- **Key technical components:**
    1. **Encoder:** SE(3)-equivariant GNN (e.g., EGNN, PaiNN) that maps atomic coordinates + features → latent state $z_t$.
    2. **Dynamics model:** A score-based diffusion model or latent ODE/SDE that evolves $z_t \rightarrow z_{t+1}$ conditioned on external actions (temperature, pressure, applied force, ligand perturbation).
    3. **Decoder:** Reconstructs atomic coordinates and observables (RMSD, RMSF, contact maps) from latent states.
    4. **Reward model:** Predicts molecular properties of interest — binding free energy ($\Delta G$), thermostability ($T_m$), solvent-accessible surface area (SASA).
    5. **Policy:** Model-based RL agent (actor-critic in latent space) that proposes mutations, ligand modifications, or simulation parameters to optimise the reward.

- **Training data:** OpenMM / GROMACS trajectories from established benchmarks (ATLAS protein dynamics dataset, DE Shaw BPTI/ubiquitin datasets, PDBBind for protein–ligand complexes). Can also bootstrap from coarse-grained MD (Martini 3) for faster data generation.

- **Concrete deliverable:** An open-source framework (`molworld`) where a user defines a molecular environment (PDB + force field), trains a latent world model on MD trajectory data, and then uses model-based RL to optimise molecular properties (binding affinity, fold stability, conformational transition rates) — all "in imagination" with orders-of-magnitude speedup over running explicit MD.

- **Potential impact:** Enables virtual high-throughput screening of mutations and drug candidates at a fraction of the computational cost, democratising long-timescale MD insights for labs without supercomputer access.

---

### ProteinDreamer — Model-Based RL for Protein Design

- **The big idea:** Frame protein design as a sequential decision-making (RL) problem where the "environment" is the protein fitness landscape, and train a world model of that landscape so the agent can plan multi-step design trajectories without querying expensive oracles (wet-lab assays, AlphaFold inference, or MD simulations) at every step.

    - **State** $s_t$ = current protein sequence $\mathbf{x} \in \{A, C, \ldots, Y\}^L$ and its predicted/known structure (backbone coordinates, pLDDT, contact map).
    - **Action** $a_t$ = a discrete mutation (single-site substitution, insertion, deletion) or a structured edit (loop redesign, domain swap) on the sequence.
    - **Transition** $s_{t+1} = f_\theta(s_t, a_t)$ = the world model's prediction of how the structure and properties change after the edit.
    - **Reward** $r_t$ = predicted fitness score — thermostability ($\Delta\Delta G$), binding affinity ($K_d$), catalytic activity ($k_{cat}$), expressibility, or a multi-objective combination.

- **The world model learns to predict:** Given a current sequence–structure pair and a proposed mutation → the resulting structure perturbation + fitness change. This allows the agent to *dream* entire evolutionary trajectories (chains of mutations) and evaluate them before committing to expensive ground-truth evaluation.

- **Why it's novel and timely:**
    - Existing generative protein design methods (ProteinMPNN, RFdiffusion, Chroma, EvoDiff, ESM-3) are predominantly **one-shot**: they generate a single sequence or structure in one forward pass, with no iterative refinement loop and no explicit planning over multi-step mutation paths.
    - Directed evolution in the wet lab *is* sequential search, but current computational tools don't model it as such. **ProteinDreamer bridges this gap** by providing an in-silico directed evolution engine guided by a learned world model.
    - The world model can be **fine-tuned online** with a small number of real experimental measurements (active learning), creating a tight compute–experiment loop suitable for real-world protein engineering campaigns.

- **Theoretical backbone — Active Inference / Free Energy Principle:**
    - Draw from the neuroscience-rooted framework of *predictive coding* and *active inference* (Friston, 2010): the agent maintains an internal generative model of the protein fitness landscape and selects actions (mutations) that minimise *expected free energy* — simultaneously seeking reward (exploitation) and reducing model uncertainty (exploration).
    - This provides a principled way to balance exploration vs. exploitation in sequence space, which is a core challenge in protein engineering (the fitness landscape is vast, rugged, and only partially observed).
    - Theoretical novelty: formalising protein design under the Free Energy Principle connects computational neuroscience, Bayesian inference, RL, and molecular biology in a unified mathematical framework.

- **Key technical components:**
    1. **Protein encoder:** Pre-trained protein language model (ESM-2, ProGen2) for sequence embeddings + structure encoder (GVP-GNN, ProteinMPNN encoder) for geometric features → joint latent state $z_t$.
    2. **World model:** Conditional latent diffusion model or autoregressive transformer that predicts $z_{t+1}$ given $(z_t, a_t)$.
    3. **Fitness predictor (reward head):** Multi-task head predicting stability, binding, function scores from $z_t$. Trained on experimental fitness landscape datasets.
    4. **Policy / planner:** Latent-space actor-critic (SAC / PPO variant) or Monte Carlo Tree Search (MCTS) over mutation trajectories in the world model.
    5. **Uncertainty module:** Ensemble or evidential deep learning for epistemic uncertainty — guides active learning and exploration.

- **Training data:** Deep mutational scanning (DMS) datasets (ProteinGym benchmark — 200+ assays), mega-scale stability data (Tsuboyama et al., 2023), AlphaFold2/ESMFold predicted structures as cheap oracles, enzyme activity datasets (BRENDA, EnzML).

- **Concrete deliverable:** `protein-dreamer` — a framework for model-based RL protein design where a user specifies a wild-type protein + target property, and the system dreams optimal mutation paths, ranks candidates, and optionally interfaces with a wet-lab active learning loop.

- **Potential impact:** Could fundamentally change how protein engineering is done — replacing random mutagenesis + screening with intelligent, model-guided, iterative design. Directly applicable to enzyme engineering, therapeutic antibody optimisation, and vaccine antigen design.

---

### CellDreamer — World Models for Gene Regulatory Network Dynamics

- **The big idea:** Instead of hand-crafting ODE/PDE reaction-diffusion models for gene regulatory networks (GRNs), learn a world model of cellular dynamics directly from data — single-cell RNA-seq time series, Perturb-seq experiments, or spatial transcriptomics atlases. The world model captures how the transcriptomic state of a cell evolves over time and in response to perturbations, enabling *in-silico* virtual cell experiments.

- **Why it's novel:**
    - Existing computational approaches to GRN modelling are either mechanistic (Boolean networks, ODEs — scalability-limited) or purely correlational (GENIE3, GRNBoost2 — no dynamics). Recent deep learning models for perturbation prediction (GEARS, CPA, scGPT) predict steady-state outcomes but **don't model the temporal trajectory**.
    - CellDreamer frames cellular dynamics as a **sequential environment**: the agent observes a cell state, proposes an intervention, and the world model predicts the resulting temporal trajectory of gene expression — enabling long-horizon planning over multi-gene, multi-timepoint perturbation strategies.

- **Key technical components:**
    1. **Latent dynamics model:** Neural ODE or SDE in a learned latent space (VAE-based encoder from scRNA-seq profiles) that evolves cell state $z_t \rightarrow z_{t+\Delta t}$! Alternatively, a conditional diffusion model over gene expression space.
    2. **Perturbation-conditioned transitions:** The dynamics model is conditioned on interventions — gene knockouts (CRISPRi/a), drug treatments (dose + time), or combinations. This is the "action" in the RL formulation.
    3. **Reward model:** User-defined cellular objective — e.g., drive a cancer cell toward apoptosis, reprogram a fibroblast into a cardiomyocyte, maximise production of a target metabolite.
    4. **Policy:** RL agent that plans optimal perturbation sequences (which genes to knock out/activate, which drugs to apply, in what order and timing) to achieve the desired cell fate transition.

- **Training data:** Perturb-seq datasets (Replogle et al., 2022 — genome-wide CRISPRi in K562/RPE1), Norman et al. (2019 — combinatorial CRISPRa), sci-Plex (drug perturbations), spatial transcriptomics time-courses (Stereo-seq, MERFISH developmental atlases).

- **Concrete deliverable:** `cell-dreamer` — a framework for learning world models of cellular dynamics from perturbation data, with an RL-based intervention planner. Users provide a perturbation dataset + target cell state, and the system designs multi-step intervention strategies in silico.

- **Potential impact:** Enables virtual perturbation experiments at scale — predicting the outcome of combinatorial gene knockouts or drug cocktails without running every combination in the lab. Directly applicable to drug discovery (identifying synergistic drug combinations), cell therapy (optimising reprogramming protocols), and synthetic biology (designing genetic circuits).
