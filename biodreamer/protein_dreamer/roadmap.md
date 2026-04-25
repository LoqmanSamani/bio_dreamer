# ProteinDreamer — Development Roadmap

## Introduction

### The Problem

Protein engineering navigates a sequence space of $20^L$ possibilities (for a protein of length $L$) over a rugged, epistatic fitness landscape that is only partially observed. The two dominant computational strategies each leave something on the table:

- **Generative protein design** (ProteinMPNN, RFdiffusion, ESM-3) produces candidate sequences in a single forward pass — powerful, but fundamentally one-shot with no mechanism for iterative refinement or look-ahead planning over sequences of mutations.
- **Directed evolution** (mutate → screen → select → repeat) is sequential by nature, but current computational tools do not model it as a sequential decision-making problem. Each round of screening is expensive, blind, and wasteful.

Neither approach does what a thoughtful engineer would do: build an internal model of how mutations reshape the fitness landscape, then plan a sequence of edits by mentally simulating their consequences before committing to the wet lab.

### The Solution: ProteinDreamer

ProteinDreamer frames protein design as a **Markov Decision Process** and applies **model-based reinforcement learning** with **latent-space world models** to plan multi-step mutation trajectories entirely in imagination.

The system:

1. **Encodes** protein states (sequence + predicted structure) into a compact latent representation using ESM-2 and GVP-GNN.
2. **Predicts** post-mutation latent states through a conditional denoising diffusion model (Latent Diffusion JEPA) or a deterministic Transformer predictor (Energy-Based JEPA) operating in latent space.
3. **Scores** latent states via a multi-task reward head predicting fitness (stability $\Delta\Delta G$, binding affinity $K_d$, catalytic activity $k_\text{cat}$).
4. **Plans** mutations via an Active Inference policy that minimizes expected free energy — naturally balancing reward-seeking (exploitation) with uncertainty reduction (exploration).

Because all planning occurs in latent space, ProteinDreamer can dream thousands of mutation trajectories efficiently without invoking expensive structure predictors or wet-lab assays at each step.

### Two World Model Architectures

- **Architecture A — Latent Diffusion JEPA (primary):** Conditional DDPM in JEPA latent space. Captures multi-modal, stochastic fitness landscapes. Built-in uncertainty from diffusion sample variance. Slower (iterative denoising).
- **Architecture B — Energy-Based JEPA (baseline):** Deterministic Transformer predictor, single forward pass. Faster for screening, but averages over modes. Uncertainty estimated externally via ensembles or evidential deep learning.

The development plan follows:

### MDP Formulation

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)
$$

- **State** $s_t = (\mathbf{x}_t, \mathbf{C}_t, \mathbf{p}_t)$: amino acid sequence, C$\alpha$ coordinates (from ESMFold), per-residue properties (pLDDT, contact maps).
- **Action** $a_t = (i, \text{AA}_\text{new})$: single-site substitution at position $i$ with amino acid $\text{AA}_\text{new}$. Action space: $\mathcal{A} = \{(i, j) : i \in \{1,\ldots,L\},\ j \in \{1,\ldots,20\}\}$.
- **Transition** $\mathcal{T}$: the learned world model $p_\theta(z_{t+1} | z_t, a_t)$.
- **Reward** $\mathcal{R}(z_t)$: predicted fitness from the reward head.
- **Discount** $\gamma = 0.99$.

---

## Development Steps

### Step 1: Base Classes in `biodreamer/core/`

**Why:** Every ProteinDreamer component inherits from a shared abstract interface. Implementing the core base classes first ensures a consistent API across all three BioDreamer modules (ProteinDreamer, MolDreamer, CellDreamer), enables code reuse, and guarantees that the protein-specific implementations conform to a shared contract.

**Where:** Each base class in its own file inside `biodreamer/core/`.

**Base classes needed:** Yes — these *are* the base classes.

**Structure:**

#### 1a. `BaseEncoder` → `biodreamer/core/encoder.py`

Abstract base class for all observation-to-latent encoders.

```
BaseEncoder(nn.Module):
    @abstractmethod encode(observation) → z_t ∈ ℝ^d
    @abstractmethod get_latent_dim() → int
```

#### 1b. `BaseDynamics` → `biodreamer/core/dynamics.py`

Abstract base for latent-space transition models.

```
BaseDynamics(nn.Module):
    @abstractmethod predict(z_t, action_emb) → ẑ_{t+1}
    predict_distribution(z_t, action_emb, n_samples) → {ẑ_{t+1}^(n)}
    rollout(z_0, actions, horizon) → [ẑ_1, ..., ẑ_H]
```

#### 1c. `BaseDecoder` → `biodreamer/core/decoder.py`

Abstract base for latent-to-observation decoders (used for validation/interpretability, not during planning).

```
BaseDecoder(nn.Module):
    @abstractmethod decode(z_t) → observation
    decode_batch(z_batch) → observations
```

#### 1d. `BaseRewardHead` → `biodreamer/core/reward.py`

Abstract base for latent-state fitness prediction.

```
BaseRewardHead(nn.Module):
    @abstractmethod predict(z_t) → scalar reward
    predict_multi(z_t) → dict of per-objective rewards
```

#### 1e. `BasePolicy` → `biodreamer/core/policy.py`

Abstract base for mutation-selection policies.

```
BasePolicy(nn.Module):
    @abstractmethod select_action(z_t) → action
    select_action_with_exploration(z_t) → action
    get_action_distribution(z_t) → distribution
    @abstractmethod update(batch) → loss_dict
```

#### 1f. `WorldModel` → `biodreamer/core/world_model.py`

Composition class that wires encoder + dynamics + decoder + reward into a unified interface.

```
WorldModel(nn.Module):
    __init__(encoder, dynamics, decoder, reward_head)
    encode(observation) → z_t
    imagine(z_t, actions) → trajectory of (ẑ, r̂) pairs
    decode(z_t) → observation
    predict_reward(z_t) → reward
```

#### 1g. `ExpectedFreeEnergy` + `ActiveInferencePolicy` → `biodreamer/core/active_inference.py`

Active Inference components: expected free energy computation and policy wrapper.

$$
G(\pi) = \underbrace{\mathbb{E}\left[\sum_{t=1}^{H} -r_t\right]}_{\text{Pragmatic value}} + \underbrace{\mathbb{E}\left[\sum_{t=1}^{H} -\eta \cdot \text{Unc}(z_t, a_t)\right]}_{\text{Epistemic value}}
$$

```
ExpectedFreeEnergy:
    compute_efe(rewards, uncertainties, eta) → G(π)
    pragmatic_value(rewards) → scalar
    epistemic_value(uncertainties, eta) → scalar

ActiveInferencePolicy(BasePolicy):
    select_action(z_t) → action minimizing G(π)
    update(imagined_batch) → loss_dict
```

**Frontend/API:** None. These are internal abstractions only.

---

### Step 2: Data Loading & Preprocessing — `biodreamer/protein_dreamer/data/`

**Why:** Before any model can be trained, we need structured PyTorch datasets that load DMS fitness data (ProteinGym, Tsuboyama), tokenize sequences, parse mutation strings, and optionally predict/load 3D structures via ESMFold. This is the foundation everything else builds on.

**Where:**`biodreamer/protein_dreamer/data/dataset.py` and `biodreamer/protein_dreamer/data/preprocessing.py`.

**Base class needed:** No — these are data utilities, not model components.

**Structure:**

#### 2a. Preprocessing utilities → `data/preprocessing.py`

Functions for converting raw data into model-ready tensors:

- `tokenize_sequence(seq: str) → Tensor`: Convert amino acid string to integer token IDs (ESM-2 tokenizer).
- `parse_mutation_string(mut_str: str) → list[tuple]`: Parse ProteinGym-format mutation strings (e.g., `"A42G:K56R"`) into `[(wt_aa, pos, mut_aa), ...]`.
- `encode_mutation(position, wt_aa, mut_aa) → Tensor`: Create the mutation action tensor combining positional encoding + AA identity embeddings.
- `load_structure(pdb_path) → Tensor`: Load PDB/mmCIF and extract C$\alpha$ coordinates as `(L, 3)` tensor.
- `predict_structure_esmfold(sequence) → Tensor`: Run ESMFold to get predicted C$\alpha$ coordinates.
- `build_protein_graph(coords, cutoff=10.0) → Data`: Build a PyG graph with residue nodes and distance-based edges for GVP-GNN.

#### 2b. Datasets → `data/dataset.py`

PyTorch Dataset classes that return $(s_t, a_t, r_t, s_{t+1})$ transition tuples:

- `ProteinGymDataset`: Loads a single ProteinGym DMS assay CSV. Each sample is a (wild-type → mutant) transition with the DMS fitness score as reward.
- `TsuboyamaDataset`: Loads the mega-scale $\Delta G$ dataset (776k+ measurements). Returns stability values.
- `FitnessTransitionDataset`: Wraps any DMS dataset to produce consecutive-mutation transition pairs for world model training.

**Frontend/API:** None — data loading is backend infrastructure. However, `scripts/download_data.sh` should be extended to fetch ProteinGym and Tsuboyama datasets.

---

### Step 3: Protein Encoder — `biodreamer/protein_dreamer/encoder.py`

**Why:** The encoder is the entry point of the world model — it maps raw protein observations (sequence + structure) into the compact latent space $z_t \in \mathbb{R}^d$ where all downstream components operate. Without a working encoder, no other component can function.

**Where:** `ProteinEncoder` class developed inside `biodreamer/protein_dreamer/encoder.py`.

**Base class:** Yes — inherits from `BaseEncoder` (developed in Step 1a in `biodreamer/core/encoder.py`).

**Structure:**

The encoder has two complementary branches fused into a single latent vector:

**Sequence branch (ESM-2):**

$$
\mathbf{H}^\text{seq} = \text{ESM-2}(\mathbf{x}_t) \in \mathbb{R}^{L \times 1280}, \qquad \mathbf{h}^\text{seq} = \text{MeanPool}(\mathbf{H}^\text{seq}) \in \mathbb{R}^{1280}
$$

**Structure branch (GVP-GNN):**

$$
\mathbf{H}^\text{struct} = \text{GVP-GNN}(\mathbf{C}_t) \in \mathbb{R}^{L \times d_\text{gvp}}
$$

GVP message passing on protein graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

$$
\mathbf{m}_{ij} = \text{GVP}\Big([\mathbf{s}_i \| \mathbf{s}_j \| \|\vec{r}_{ij}\|],\ [\vec{v}_i \| \vec{v}_j \| \vec{r}_{ij}]\Big)
$$

$$
\mathbf{s}_i', \vec{v}_i' = \text{GVP}\Big(\mathbf{s}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(s)},\ \vec{v}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(v)}\Big)
$$

$$
\mathbf{h}^\text{struct} = \text{MeanPool}(\mathbf{H}^\text{struct}) \in \mathbb{R}^{d_\text{gvp}}
$$

**Fusion MLP:**

$$
z_t = \text{LayerNorm}\Big(\text{MLP}\big([\mathbf{h}^\text{seq} \| \mathbf{h}^\text{struct}]\big)\Big) \in \mathbb{R}^d
$$

**SIGReg regularization** (enforces $z_t \sim \mathcal{N}(0, \mathbf{I})$): Projects the batch of embeddings onto random unit-norm directions, tests univariate normality via Epps-Pulley test, aggregates via Cramér-Wold theorem. Replaces VICReg's 6-7 hyperparameters with a single $\lambda_\text{reg}$.

**Target encoder (EMA):**

$$
\bar{\theta} \leftarrow \tau_\text{ema}\, \bar{\theta} + (1 - \tau_\text{ema})\, \theta, \qquad \bar{z}_{t+1} = f_{\bar{\theta}}(s_{t+1}) \quad \text{(stop-gradient)}
$$

```
ProteinEncoder(BaseEncoder):
    __init__(esm_model_name, gvp_hidden_dim, latent_dim, freeze_esm)
    encode_sequence(tokens) → h_seq
    encode_structure(graph) → h_struct
    fuse(h_seq, h_struct) → z_t
    encode(observation) → z_t       # full pipeline
    get_latent_dim() → int
```

Also implement:

- `ActionEncoder` (small MLP): $a_t^\text{emb} = e_\alpha(a_t) = \text{MLP}\Big([\text{PosEmb}(i) \| \text{AAEmb}(\text{AA}_\text{old}) \| \text{AAEmb}(\text{AA}_\text{new})]\Big) \in \mathbb{R}^{d_a}$

**Frontend/API:** None at this stage. The encoder is an internal model component.

---

### Step 4: Dynamics Model — `biodreamer/protein_dreamer/dynamics.py`

**Why:** The dynamics model is the core of the world model — it predicts how the latent state changes after a mutation. Without it, there is no "dreaming." This is where Architecture A (Latent Diffusion JEPA) and Architecture B (Energy-Based JEPA) diverge.

**Where:** `ProteinDynamics` class inside `biodreamer/protein_dreamer/dynamics.py`.

**Base class:** Yes — inherits from `BaseDynamics` (Step 1b in `biodreamer/core/dynamics.py`).

**Structure:**

#### Architecture B — Energy-Based JEPA (implement first)

Deterministic Transformer predictor, single forward pass:

$$
\hat{z}_{t+1} = g_\phi(z_t, a_t^\text{emb}) = \text{TransformerBlock}\big([z_t \| a_t^\text{emb}]\big)[:d]
$$

Training loss:

$$
\mathcal{L}_B = \beta_\text{jepa} \cdot \underbrace{\frac{1}{B}\sum_{i=1}^{B} \|g_\phi(z_t^{(i)}, a_t^{(i)\text{emb}}) - \text{sg}(\bar{z}_{t+1}^{(i)})\|_2^2}_{\mathcal{L}_\text{pred}} + \lambda_\text{reg} \cdot \text{SIGReg}(z_t) + \beta_\text{rew} \cdot \mathcal{L}_\text{rew}
$$

#### Architecture A — Latent Diffusion JEPA (implement second)

Conditional DDPM operating in JEPA latent space.

**Forward process** (cosine noise schedule, $K$ steps):

$$
\bar{z}_{t+1}^{\tau} = \sqrt{\bar{\alpha}_\tau}\, \bar{z}_{t+1} + \sqrt{1 - \bar{\alpha}_\tau}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I}), \quad \bar{\alpha}_\tau = \prod_{i=1}^{\tau} \alpha_i
$$

**Denoiser** ($x_0$-prediction):

$$
\hat{z}_{t+1}^0 = D_{\hat{\theta}}\big(\bar{z}_{t+1}^{\tau},\ \tau,\ z_t,\ a_t^\text{emb}\big)
$$

**Reverse process:**

$$
\hat{z}_{t+1}^{\tau-1} = \frac{\sqrt{\bar{\alpha}_{\tau-1}}\,\beta_\tau}{1 - \bar{\alpha}_\tau}\,D_{\hat{\theta}}(\hat{z}_{t+1}^{\tau}, \tau, z_t, a_t^\text{emb}) + \frac{\sqrt{\alpha_\tau}\,(1 - \bar{\alpha}_{\tau-1})}{1 - \bar{\alpha}_\tau}\,\hat{z}_{t+1}^{\tau} + \sigma_\tau\,\mathbf{w}
$$

**Training loss:**

$$
\mathcal{L}_A = \beta_\text{diff} \cdot \underbrace{\mathbb{E}_{\tau, \varepsilon}\Big[\|D_{\hat{\theta}}(\bar{z}_{t+1}^{\tau}, \tau, z_t, a_t^\text{emb}) - \bar{z}_{t+1}\|_2^2\Big]}_{\mathcal{L}_\text{diff}} + \lambda_\text{reg} \cdot \text{SIGReg}(z_t) + \beta_\text{rew} \cdot \mathcal{L}_\text{rew}
$$

```
EnergyBasedDynamics(BaseDynamics):
    __init__(latent_dim, action_dim, num_layers, num_heads)
    predict(z_t, action_emb) → ẑ_{t+1}

DiffusionDynamics(BaseDynamics):
    __init__(latent_dim, action_dim, diffusion_steps, noise_schedule)
    noise(z_clean, tau) → z_noised
    denoise(z_noised, tau, z_t, action_emb) → ẑ_clean
    predict(z_t, action_emb) → ẑ_{t+1}   # full reverse diffusion
    predict_distribution(z_t, action_emb, n_samples) → {ẑ_{t+1}^(n)}

MutationEmbedding(nn.Module):
    __init__(max_seq_len, num_amino_acids, embed_dim)
    forward(position, wt_aa, mut_aa) → action_emb
```

**Frontend/API:** None at this stage.

---

### Step 5: Reward Head — `biodreamer/protein_dreamer/reward.py`

**Why:** The reward head translates latent states into fitness predictions — it is what grounds the latent space in experimentally measurable protein properties. Without it, the agent has no objective to optimize. It also provides the supervised signal ($\mathcal{L}_\text{rew}$) that shapes the latent space during world model training.

**Where:** `ProteinRewardHead` class inside `biodreamer/protein_dreamer/reward.py`.

**Base class:** Yes — inherits from `BaseRewardHead` (Step 1d in `biodreamer/core/reward.py`).

**Structure:**

Multi-task MLP with shared backbone and task-specific output heads:

$$
\hat{r}_t = R_\psi(z_t) = \big[\hat{y}^\text{stab}_t,\ \hat{y}^\text{bind}_t,\ \hat{y}^\text{act}_t\big]
$$

$$
\mathcal{L}_\text{rew} = \sum_{k} w_k \cdot \text{SmoothL1}\big(\hat{y}^{(k)}_t, y^{(k)}_t\big)
$$

```
ProteinRewardHead(BaseRewardHead):
    __init__(latent_dim, num_objectives, hidden_dims, task_weights)
    shared_backbone: MLP [latent_dim → hidden → hidden]
    stability_head:  Linear [hidden → 1]      # ΔΔG
    affinity_head:   Linear [hidden → 1]      # Kd
    activity_head:   Linear [hidden → 1]      # kcat

    predict(z_t) → scalarized reward
    predict_multi(z_t) → {stability, affinity, activity}
    compute_loss(z_t, targets) → L_rew
```

**Scalarization strategies:**

- Weighted sum: $r = \sum_k w_k \hat{y}_k$
- Tchebycheff (Pareto-based): $r = \min_k |y_k^\text{target} - \hat{y}_k|$

**Frontend/API:** None at this stage. Reward predictions will later be exposed through the inference pipeline.

---

### Step 6: Uncertainty Module — `biodreamer/protein_dreamer/uncertainty.py`

**Why:** Uncertainty estimation is essential for the Active Inference policy's epistemic value term — it tells the agent where the world model is unreliable, directing exploration toward the most informative mutations. It also safeguards against model exploitation during policy training.

**Where:** `UncertaintyModule` and strategy classes inside `biodreamer/protein_dreamer/uncertainty.py`.

**Base class:** No dedicated base class in `biodreamer/core/` — uncertainty is protein_dreamer-specific for now. However, it interacts closely with `ExpectedFreeEnergy` in `biodreamer/core/active_inference.py`.

**Structure:**

**Architecture A** — built-in from $N$ diffusion samples:

$$
\text{Unc}(z_t, a_t) = \frac{1}{N}\sum_{n=1}^N \|\hat{z}_{t+1}^{(n)} - \bar{\hat{z}}_{t+1}\|_2^2, \qquad \bar{\hat{z}}_{t+1} = \frac{1}{N}\sum_{n=1}^N \hat{z}_{t+1}^{(n)}
$$

**Architecture B** — ensemble of $M$ predictors:

$$
\text{Unc}_\text{ens}(z_t, a_t) = \frac{1}{M}\sum_{m=1}^M \|g_{\phi_m}(z_t, a_t^\text{emb}) - \bar{g}(z_t, a_t^\text{emb})\|_2^2
$$

```
UncertaintyModule:
    __init__(strategy: str)  # "diffusion_variance" | "ensemble" | "evidential" | "mc_dropout"

EnsembleUncertainty:
    __init__(dynamics_model, num_members)
    estimate(z_t, action_emb) → uncertainty_scalar

EvidentialUncertainty:
    __init__(latent_dim)
    estimate(z_t) → (mu, sigma, alpha, beta)   # Normal-Inverse-Gamma parameters

MCDropoutUncertainty:
    __init__(model, num_samples, dropout_rate)
    estimate(z_t, action_emb) → uncertainty_scalar
```

**Frontend/API:** None. Uncertainty feeds into the policy's EFE computation internally.

---

### Step 7: Protein Environment — `biodreamer/protein_dreamer/environment.py`

**Why:** The environment provides the Gym-like interface that connects the world model to training and evaluation loops. It wraps fitness oracles (DMS lookup, ESMFold proxy, or wet-lab interface) and manages the protein state across mutation steps. The policy trains on imagined rollouts from the world model, but the environment is needed for (a) collecting initial training data, (b) evaluating designed candidates, and (c) the active learning loop.

**Where:** `ProteinEnvironment` class inside `biodreamer/protein_dreamer/environment.py`.

**Base class:** No — environments are not model components, so no `biodreamer/core/` class is needed. Follows the standard Gymnasium API convention.

**Structure:**

```
ProteinEnvironment:
    __init__(wild_type_sequence, oracle, max_mutations, encoder)
    reset() → (observation, info)
    step(mutation_action) → (observation, reward, terminated, truncated, info)
    render() → visualization dict

Fitness Oracle Options:
    DMSLookupOracle:   Direct lookup from experimental DMS dataset
    ESMFoldOracle:     Proxy fitness via ESMFold pLDDT + ProteinMPNN
    PredictorOracle:   Pre-trained fitness predictor (Tranception, ESM-2 zero-shot)
    WetLabOracle:      Interface for real experimental feedback (active learning)
```

**Frontend/API:** The environment's step/render interface will later be exposed through the server's `/protein_dreamer/dream` endpoint for interactive exploration.

---

### Step 8: Policy — `biodreamer/protein_dreamer/policy.py`

**Why:** The policy is the agent that actually selects mutations — it is the decision-maker that turns world model predictions into actionable protein designs. Multiple policy implementations are needed to compare RL strategies and validate that Active Inference provides a principled advantage.

**Where:** Policy classes inside `biodreamer/protein_dreamer/policy.py`.

**Base class:** Yes — all inherit from `BasePolicy` (Step 1e) and `ActiveInferencePolicy` (Step 1g) in `biodreamer/core/`.

**Structure:**

**Active Inference policy (primary):**

$$
G(\pi) = \underbrace{\mathbb{E}_{q_\phi}\Big[\sum_{t=1}^{H} -r_t\Big]}_{\text{Pragmatic value}} + \underbrace{\mathbb{E}_{q_\phi}\Big[\sum_{t=1}^{H} -\eta \cdot \text{Unc}(z_t, a_t)\Big]}_{\text{Epistemic value}}, \qquad \pi^* = \arg\min_\pi G(\pi)
$$

**Actor update** (policy gradient):

$$
\nabla_\xi J = \mathbb{E}\Big[\nabla_\xi \log \pi_\xi(a_t | z_t) \cdot (-G_t)\Big]
$$

**Critic update** (TD($\lambda$) on imagined trajectories):

$$
\mathcal{L}_\text{critic} = \mathbb{E}\Big[(V_\omega(z_t) - V_t^{\lambda_\text{td}})^2\Big], \quad V_t^{\lambda_\text{td}} = (1-\lambda_\text{td})\sum_{n=1}^{H-t} \lambda_\text{td}^{n-1} V_t^{(n)}
$$

```
ProteinActiveInferencePolicy(ActiveInferencePolicy):
    __init__(latent_dim, action_dim, eta, horizon)
    actor: MLP [latent_dim → L×20 action logits]
    critic: MLP [latent_dim → scalar value]
    select_action(z_t) → mutation minimizing G(π)
    imagine_and_update(world_model, z_t) → loss_dict

ProteinPPO(BasePolicy):
    __init__(latent_dim, action_dim, clip_ratio, entropy_coeff)
    select_action(z_t) → mutation
    update(trajectories) → loss_dict

ProteinMCTS(BasePolicy):
    __init__(world_model, reward_head, num_simulations, exploration_weight)
    select_action(z_t) → best mutation from tree search
    search(z_t, depth) → mutation tree
```

**Frontend/API:** Policy selection will be configurable via the API request payload (e.g., `"policy": "active_inference"` vs `"ppo"`).

---

### Step 9: Decoder — `biodreamer/protein_dreamer/decoder.py`

**Why:** The decoder reconstructs interpretable outputs (amino acid probabilities, 3D coordinates, pLDDT scores) from latent states. It is NOT used during planning (the JEPA paradigm discards the decoder to gain speed). It serves two purposes: (a) interpretability — showing users what mutations the agent proposes, and (b) validation — checking world model latent states against high-fidelity structure predictors.

**Where:** `ProteinDecoder` class inside `biodreamer/protein_dreamer/decoder.py`.

**Base class:** Yes — inherits from `BaseDecoder` (Step 1c in `biodreamer/core/decoder.py`).

**Structure:**

```
ProteinDecoder(BaseDecoder):
    __init__(latent_dim, seq_length, num_amino_acids)
    decode_sequence(z_t) → (L, 20)   # per-position AA probabilities
    decode_structure(z_t) → (L, 3)   # predicted Cα coordinates
    decode_plddt(z_t) → (L,)         # per-residue confidence
    decode(z_t) → dict               # all outputs combined
```

**Frontend/API:** Decoded sequences and structures are critical for the frontend's Mol* 3D viewer and results display. The decoder output will be served through the `/protein_dreamer/dream` response.

---

### Step 10: Training Infrastructure — `biodreamer/training/`

**Why:** With all model components in place, we need the training loops that coordinate world model training, policy training, and active learning. Training proceeds in two alternating phases (as specified in the PhD proposal): world model updates on DMS data, and policy updates on imagined rollouts.

**Where:** Training classes inside `biodreamer/training/`.

**Base class:** `Trainer` is its own base class; specific trainers for world model and policy inherit from it.

**Structure:**

#### 10a. `WorldModelTrainer` → `biodreamer/training/world_model_trainer.py`

Trains encoder + dynamics + reward head jointly. Implements Algorithm 1 from the proposal:

1. Sample batch $(s_t, a_t, r_t, s_{t+1})$ from DMS data.
2. Encode: $z_t = f_\theta(s_t)$, $\bar{z}_{t+1} = f_{\bar{\theta}}(s_{t+1})$ (stop-gradient).
3. Compute combined loss $\mathcal{L}_A$ or $\mathcal{L}_B$.
4. Update encoder, dynamics, reward head.
5. EMA update target encoder: $\bar{\theta} \leftarrow \tau_\text{ema}\, \bar{\theta} + (1 - \tau_\text{ema})\, \theta$.

#### 10b. `PolicyTrainer` → `biodreamer/training/policy_trainer.py`

Trains actor + critic on imagined rollouts (Algorithm 2):

1. Sample initial $z_0$ from encoded replay buffer.
2. Rollout $H$ steps in world model, collecting $(z_t, a_t, r_t, u_t)$.
3. Compute TD($\lambda$) returns.
4. Update critic and actor.

#### 10c. `ActiveLearningLoop` → `biodreamer/training/active_learning.py`

Dream → Propose → Evaluate → Update cycle:

1. Dream candidates using policy + world model.
2. Select top-$k$ candidates by EFE (balancing fitness and uncertainty).
3. Evaluate candidates with oracle (DMS lookup or wet lab).
4. Fine-tune world model on new data.
5. Repeat.

#### 10d. Callbacks & Schedulers

- `biodreamer/training/callbacks.py`: W&B logging, checkpointing, early stopping.
- `biodreamer/training/schedulers.py`: WarmupCosine, CyclicKL, KLBalance LR schedules.

**Frontend/API:** Training jobs will be submitted and monitored through the server's `/jobs/` endpoints and the frontend's job tracking page.

---

### Step 11: Inference Pipeline — `biodreamer/inference/`

**Why:** The inference pipeline is the end-user-facing system — it takes a wild-type protein, runs the trained world model + policy to dream mutation trajectories, ranks candidates, and returns designed sequences. This is Algorithm 3 from the proposal.

**Where:** Inference classes inside `biodreamer/inference/`.

**Base class:** No — these are orchestration classes, not model components.

**Structure:**

#### 11a. `Dreamer` → `biodreamer/inference/dreamer.py`

Imagination engine: encodes wild-type, rolls out $C$ candidate trajectories of $T$ mutation steps each, collects predicted fitness at each step.

#### 11b. `CandidateRanker` → `biodreamer/inference/candidate_ranker.py`

Ranks designed candidates by predicted fitness, diversity (sequence edit distance), and optionally Pareto optimality for multi-objective designs.

#### 11c. `ProteinDreamerPipeline` → `biodreamer/inference/pipeline.py`

End-to-end pipeline: input sequence → encode → dream $C$ trajectories → rank → return top-$k$ designed sequences with fitness predictions and uncertainty estimates.

**Frontend/API:** This is the primary interface to the system. Exposed as:

- **Server:** `POST /protein_dreamer/dream` in `server/routers/protein_dreamer.py`.
- **Schema:** `ProteinDesignRequest` / `DreamResult` in `server/schemas/protein.py`.
- **Service:** `ProteinService` in `server/services/protein_service.py`.
- **Frontend:** Protein Dreamer page at `/protein-dreamer` with input form, 3D viewer (Mol*), and results table.

---

### Step 12: Configuration — `configs/protein_dreamer/`

**Why:** All hyperparameters — model architecture, training schedule, data paths — should be externalized into YAML configs rather than hardcoded, enabling reproducible experiments and easy ablation studies.

**Where:** `configs/protein_dreamer/default.yaml` (and optionally per-experiment overrides).

**Base class:** N/A.

**Structure:**

Key config sections:

- `data:` — dataset paths, train/val splits, preprocessing options
- `encoder:` — ESM-2 model name, GVP-GNN dimensions, latent_dim, freeze_esm
- `dynamics:` — architecture (energy_based / diffusion), num_layers, diffusion_steps, noise_schedule
- `decoder:` — hidden_dims, output heads
- `reward:` — num_objectives, task_weights, hidden_dims
- `policy:` — type (active_inference / ppo / mcts), eta, horizon, clip_ratio
- `uncertainty:` — strategy (ensemble / evidential / mc_dropout), num_members
- `training:` — batch_size, lr, ema_decay, num_epochs, gradient_clip
- `active_learning:` — budget, acquisition_batch_size, retrain_interval
- `logging:` — wandb_project, log_interval, checkpoint_interval

**Frontend/API:** None — configs are developer-facing.

---

### Step 13: Hub Integration — `biodreamer/hub/`

**Why:** Trained models (encoder checkpoints, dynamics weights, reward head) should be versioned and shared via Hugging Face Hub, enabling reproducibility and community contribution. The hub module handles downloading pre-trained models, uploading new ones, and auto-generating model cards.

**Where:** `biodreamer/hub/` — `registry.py`, `download.py`, `upload.py`, `model_card.py`.

**Base class:** No — these are utility/infrastructure classes.

**Structure:**

```
ModelRegistry:
    list_models(task="protein_dreamer") → list of model cards
    get_model(model_id) → model checkpoint + config

download_from_hub(model_id, cache_dir) → local path
upload_to_hub(model, model_card, repo_id, token) → URL
generate_model_card(model, config, metrics) → ModelCard
```

**Frontend/API:**

- **Server:** `GET/POST /models/` endpoints in `server/routers/models.py`.
- **Frontend:** Model browser page at `/models` with download/upload UI.

---

### Step 14: Server & API Endpoints — `server/`

**Why:** The FastAPI server exposes the trained models to end users through a REST API. It handles request validation, job queuing (for long-running GPU inference), and serves results to the frontend.

**Where:** `server/routers/protein_dreamer.py`, `server/schemas/protein.py`, `server/services/protein_service.py`, `server/workers/gpu_worker.py`.

**Base class:** No — the server layer is separate from the ML framework.

**Structure:**

- `POST /protein_dreamer/dream`: Accept a wild-type sequence + config, queue a dreaming job, return job ID.
- `GET /protein_dreamer/jobs/{job_id}`: Poll job status and retrieve results.
- `POST /protein_dreamer/evaluate`: Score a set of candidate sequences against an oracle.

**Frontend/API:** This IS the API layer. Must be developed in coordination with the frontend (Step 15).

---

### Step 15: Frontend — `frontend/`

**Why:** The web interface makes ProteinDreamer accessible to non-programmer biologists and provides rich visualizations (3D protein structures, fitness landscapes, mutation trajectories) that are essential for interpreting results.

**Where:** `frontend/src/app/protein-dreamer/` and `frontend/src/components/protein-dreamer/`.

**Base class:** N/A (React components, not Python classes).

**Structure:**

- **Input form:** Paste wild-type sequence, select objectives, configure planning horizon.
- **3D viewer:** Mol* component for visualizing wild-type and designed structures with mutation highlights.
- **Results table:** Ranked candidates with fitness predictions, uncertainty, and trajectory visualization.
- **Fitness landscape plot:** 2D projection of the latent space showing explored regions and fitness gradients.

**Frontend/API:** This IS the frontend. Consumes the server API from Step 14.

---

### Step 16: Tests — `tests/`

**Why:** Every component needs unit tests (for correctness), integration tests (for component interaction), and e2e tests (for the full pipeline). Testing is especially critical for scientific software where silent numerical errors can invalidate results.

**Where:** `tests/unit/`, `tests/integration/`, `tests/e2e/`.

**Base class:** N/A.

**Structure:**

- `tests/unit/test_encoder.py`: Verify encoder output shapes, latent dim, SIGReg normalization.
- `tests/unit/test_dynamics.py`: Verify prediction shapes, diffusion forward/reverse consistency.
- `tests/unit/test_reward.py`: Verify multi-task output, loss computation.
- `tests/unit/test_policy.py`: Verify action sampling, gradient flow.
- `tests/integration/test_world_model.py`: Verify encode → predict → reward end-to-end.
- `tests/integration/test_training.py`: Verify one training step runs without errors.
- `tests/e2e/test_dreaming.py`: Verify full pipeline: sequence → dream → ranked candidates.

**Frontend/API:** None — tests are developer-facing CI infrastructure.

---

## Development Priority & Phasing

| Phase                          | Steps                                                       | Target                            |
| ------------------------------ | ----------------------------------------------------------- | --------------------------------- |
| **Phase 1: Foundations** | Steps 1–2 (core base classes + data loading)               | Runnable data pipeline            |
| **Phase 2: World Model** | Steps 3–5 (encoder + dynamics B + reward head)             | Trainable Energy-Based JEPA       |
| **Phase 3: Agent**       | Steps 6–8 (uncertainty + environment + policy)             | Trainable policy with imagination |
| **Phase 4: Inference**   | Steps 9–11 (decoder + training loops + inference pipeline) | End-to-end design pipeline        |
| **Phase 5: Diffusion**   | Step 4 Architecture A upgrade                               | Latent Diffusion JEPA             |
| **Phase 6: Platform**    | Steps 12–16 (config + hub + server + frontend + tests)     | Deployable application            |
