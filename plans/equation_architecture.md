# ProteinDreamer: Model-Based Reinforcement Learning for Protein Design via Latent World Models

> **PhD Proposal — Architecture Specification**
> This document defines two concrete world model architectures for ProteinDreamer, each with complete loss functions, equations, and architectural diagrams.

---

## Table of Contents

1. [Shared Foundations](#1-shared-foundations)
2. [Architecture A — Latent Diffusion JEPA (Generative)](#2-architecture-a--latent-diffusion-jepa-generative)
3. [Architecture B — Energy-Based JEPA (Non-Generative)](#3-architecture-b--energy-based-jepa-non-generative)
4. [Comparison Summary](#4-comparison-summary)

---

## 1. Shared Foundations

Both architectures share the same MDP formulation, encoder design, reward model, and Active Inference policy. They differ only in the **JEPA predictor** (world model dynamics) — Architecture A uses a conditional diffusion model in latent space; Architecture B uses a deterministic predictor with an energy-based objective.

### 1.1 MDP Formulation

Protein design is framed as a Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$:

- **State** $s_t = (\mathbf{x}_t, \mathbf{C}_t, \mathbf{p}_t)$: protein sequence $\mathbf{x}_t \in \{A, C, D, \ldots, Y\}^L$, predicted structure $\mathbf{C}_t \in \mathbb{R}^{L \times 3}$ (C$\alpha$ coordinates), and property estimates $\mathbf{p}_t \in \mathbb{R}^k$ (pLDDT, contact map, etc.).
- **Action** $a_t \in \mathcal{A}$: a mutation — single-site substitution $(i, \text{AA}_{\text{new}})$, insertion, or deletion at position $i$. The action space is $\mathcal{A} = \{(i, j) : i \in \{1,\ldots,L\}, j \in \{1,\ldots,20\}\}$ for substitutions.
- **Transition** $\mathcal{T}$: the world model predicts how the latent state changes: $z_t \xrightarrow{a_t} \hat{z}_{t+1}$.
- **Reward** $\mathcal{R}(z_t)$: predicted fitness — $\Delta\Delta G$ (stability), $K_d$ (binding), $k_{\text{cat}}$ (activity), or multi-objective.
- **Discount** $\gamma \in [0, 1]$: typically $\gamma = 0.99$ for multi-step mutation trajectories.

### 1.2 Protein Encoder (Context Encoder) $f_\theta$

The encoder maps raw protein observations to a latent state $z_t \in \mathbb{R}^d$ (where $d \approx 256$–$512$):

$$
z_t = f_\theta(s_t) = \text{FusionMLP}\Big(\text{ESM2}(\mathbf{x}_t),\; \text{GVP-GNN}(\mathbf{C}_t)\Big)
$$

**Components:**

1. **Sequence encoder — ESM-2 (650M, frozen or fine-tuned):**

   - Input: amino acid sequence $\mathbf{x}_t \in \{1,\ldots,20\}^L$
   - Output: per-residue embeddings $\mathbf{H}^{\text{seq}} \in \mathbb{R}^{L \times 1280}$
   - Global representation: $\mathbf{h}^{\text{seq}} = \text{MeanPool}(\mathbf{H}^{\text{seq}}) \in \mathbb{R}^{1280}$
2. **Structure encoder — GVP-GNN:**

   - Input: protein graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes are residues with scalar features $(s_i)$ and vector features $(\vec{v}_i)$, edges connect residues within 10Å
   - Message passing (3 layers):

$$
\mathbf{m}_{ij} = \text{GVP}\Big([\mathbf{s}_i \| \mathbf{s}_j \| \|\vec{r}_{ij}\|],\; [\vec{v}_i \| \vec{v}_j \| \vec{r}_{ij}]\Big)
$$

$$
\mathbf{s}_i', \vec{v}_i' = \text{GVP}\Big(\mathbf{s}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(s)},\; \vec{v}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(v)}\Big)
$$

   - Output: per-residue embeddings $\mathbf{H}^{\text{struct}} \in \mathbb{R}^{L \times 1280}$ (same dimension as ESM-2, for clean fusion)
   - Global representation: $\mathbf{h}^{\text{struct}} = \text{MeanPool}(\mathbf{H}^{\text{struct}}) \in \mathbb{R}^{1280}$

3. **Fusion MLP:**

$$
z_t = \text{LayerNorm}\Big(\text{MLP}_{2 \times 512}\big([\mathbf{h}^{\text{seq}} \| \mathbf{h}^{\text{struct}}]\big)\Big) \in \mathbb{R}^d, \quad [\mathbf{h}^{\text{seq}} \| \mathbf{h}^{\text{struct}}] \in \mathbb{R}^{2560}
$$

4. **SIGReg on encoder output** (replaces VICReg; LeWorldModel, Maes et al., 2026):

SIGReg (Sketched-Isotropic-Gaussian Regularizer) enforces $z_t \sim \mathcal{N}(0, \mathbf{I})$ by projecting the batch of embeddings onto $M$ random unit-norm directions, testing univariate normality on each 1D projection via the Epps-Pulley statistical test, and aggregating via the Cramér-Wold theorem (matching all 1D marginals $\Rightarrow$ matching the full joint distribution).

**Advantages over VICReg:**
- Only **1 hyperparameter** ($\lambda$) vs. VICReg's 6-7 — robust across $\lambda \in [0.01, 0.2]$
- No competing gradients from separate variance/covariance terms
- LeWorldModel showed stable end-to-end JEPA training with just prediction loss + SIGReg
- **Bonus for Architecture A:** Diffusion forward process adds $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$; if SIGReg already pushes $z_t$ toward $\mathcal{N}(0, \mathbf{I})$, the diffusion operates in a well-matched latent geometry → better denoising quality

Applied on the **context encoder output** $z_t$ only (not on predictor output or target encoder output).

### 1.3 Action Encoder $e_\alpha$

The mutation action is encoded into a continuous vector:

$$
a_t^{\text{emb}} = e_\alpha(a_t) = \text{MLP}\Big([\text{PosEmb}(i) \| \text{AAEmb}(\text{AA}_{\text{old}}) \| \text{AAEmb}(\text{AA}_{\text{new}})]\Big) \in \mathbb{R}^{d_a}
$$

where $d_a = 128$, $\text{PosEmb}$ is sinusoidal positional encoding, and $\text{AAEmb}$ is a learned amino acid embedding.

### 1.4 Target Encoder $f_{\bar{\theta}}$ (EMA)

To prevent representational collapse (a key challenge in JEPA training; LeWorldModel, Maes et al., 2026), the target latent is computed by an exponential moving average (EMA) copy of the context encoder:

$$
\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta, \quad \tau = 0.996
$$

$$
\bar{z}_{t+1} = f_{\bar{\theta}}(s_{t+1}) \quad \text{(target, stop-gradient)}
$$

The target encoder is **not** updated by gradients — only by EMA. This provides stable prediction targets for the JEPA predictor.

### 1.5 Reward Head (Fitness Predictor) $R_\psi$

A multi-task MLP predicting fitness scores from latent states:

$$
\hat{r}_t = R_\psi(z_t) = \big[\hat{y}^{\text{stab}}_t, \hat{y}^{\text{bind}}_t, \hat{y}^{\text{act}}_t\big]
$$

$$
R_\psi(z) = \text{MLP}_{3 \times 256}(z), \quad \text{with separate output heads for each fitness dimension}
$$

**Reward loss** (trained supervised on DMS data):

$$
\mathcal{L}_{\text{reward}} = \sum_{k \in \{\text{stab, bind, act}\}} w_k \cdot \text{SmoothL1}\big(\hat{y}^{(k)}_t, y^{(k)}_t\big)
$$

where $y^{(k)}_t$ is the experimental fitness measurement from ProteinGym / Tsuboyama data.

### 1.6 Active Inference Policy $\pi_\xi$

The policy selects mutations by minimising **Expected Free Energy** $G(\pi)$, which naturally decomposes into exploitation (pragmatic value) and exploration (epistemic value):

$$
G(\pi) = \underbrace{\mathbb{E}_{q_\phi}\Big[\sum_{t=1}^{H} -r_t\Big]}_{\text{Pragmatic value (exploit)}} + \underbrace{\mathbb{E}_{q_\phi}\Big[\sum_{t=1}^{H} -\mathbb{H}\big[p(z_{t+1} | z_t, a_t)\big]\Big]}_{\text{Epistemic value (explore)}}
$$

$$
\pi^* = \arg\min_\pi G(\pi)
$$

**Implementation — Latent-Space Actor-Critic:**

- **Actor** $\pi_\xi(a_t | z_t)$: outputs a categorical distribution over mutations. Trained to minimise $G(\pi)$ via policy gradient:

$$
\nabla_\xi J = \mathbb{E}\Big[\nabla_\xi \log \pi_\xi(a_t | z_t) \cdot \big(-G_t\big)\Big]
$$

- **Critic** $V_\omega(z_t)$: estimates the expected $-G(\pi)$ from state $z_t$. Trained by TD($\lambda$) on imagined trajectories:

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}\Big[\big(V_\omega(z_t) - V_t^{\lambda}\big)^2\Big]
$$

where $V_t^{\lambda} = (1-\lambda)\sum_{n=1}^{H-t} \lambda^{n-1} V_t^{(n)}$ is the $\lambda$-return computed from world model rollouts.

- **Alternative policy: MCTS.** Instead of actor-critic, use Monte Carlo Tree Search over the mutation tree. At each node, the world model simulates the mutation, the reward head scores the outcome, and UCB selects the next branch. MCTS is more sample-efficient for short horizons ($H \leq 10$ mutations).

---

## 2. Architecture A — Latent Diffusion JEPA (Generative)

### 2.1 Overview

The world model predictor is a **conditional denoising diffusion model** operating in JEPA latent space. Given the current latent state $z_t$ and mutation action $a_t$, it generates samples from the conditional distribution $p_\theta(z_{t+1} | z_t, a_t)$ via iterative denoising. This captures the **multi-modal, stochastic** nature of protein fitness landscapes.

### 2.2 Architecture Diagram

```
Architecture A — Latent Diffusion JEPA for ProteinDreamer
═══════════════════════════════════════════════════════════

                        TRAINING PHASE
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Protein State s_t          Protein State s_{t+1}                  │
│  (seq + struct)             (seq + struct after mutation)          │
│       │                            │                               │
│       ▼                            ▼                               │
│  ┌──────────┐              ┌──────────────┐                        │
│  │ Context  │              │   Target     │                        │
│  │ Encoder  │              │  Encoder     │                        │
│  │  f_θ     │              │   f_θ̄ (EMA)  │                        │
│  └────┬─────┘              └──────┬───────┘                        │
│       │                           │                                │
│       ▼                           ▼                                │
│      z_t                        z̄_{t+1}  ◄── target (stop-grad)    │
│       │                           │                                │
│       │    ┌───────────┐          │                                │
│       │    │  Action   │          │                                │
│       │    │ Encoder   │          │                                │
│       │    │  e_α(a_t) │          │                                │
│       │    └─────┬─────┘          │                                │
│       │          │                │                                │
│       ▼          ▼                ▼                                │
│  ┌─────────────────────────────────────────┐                       │
│  │         Latent Diffusion Predictor      │                       │
│  │                                         │                       │
│  │  Forward: z̄_{t+1} + noise ε ~ N(0,I)    │                       │
│  │             → z̄^τ_{t+1}                 │                       │
│  │                                         │                       │
│  │  Denoiser: D_θ(z̄^τ_{t+1}, τ, z_t, a_t)  │                       │
│  │             → ẑ^0_{t+1}                 │                       │
│  │                                         │                       │
│  │  Loss: ||ẑ^0_{t+1} - z̄_{t+1}||²         │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                            │
│       ▼                                                            │
│  Combined Loss:                                                    │
│  L = β_diff · L_diff + λ · SIGReg(Z) + β_rew · L_reward            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

                      IMAGINATION / PLANNING PHASE
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Current latent z_t                                                │
│       │                                                            │
│       │    Policy π_ξ(a|z_t) proposes mutation a_t                 │
│       │          │                                                 │
│       ▼          ▼                                                 │
│  ┌──────────────────────────────────┐                              │
│  │  Latent Diffusion Predictor      │                              │
│  │  (K denoising steps, K ≈ 10-20)  │                              │
│  │                                  │                              │
│  │  z^K ~ N(0, I)                   │                              │
│  │  for τ = K, K-1, ..., 1:         │                              │
│  │    z^{τ-1} = denoise(z^τ, τ,     │                              │
│  │              z_t, a_t)           │                              │
│  │  ẑ_{t+1} = z^0                   │                              │
│  └──────────┬───────────────────────┘                              │
│             │                                                      │
│     ┌───────┴──────────┐                                           │
│     │                  │                                           │
│     ▼                  ▼                                           │
│  ┌────────┐    ┌──────────────┐                                    │
│  │ Reward │    │ Uncertainty  │                                    │
│  │  Head  │    │ (sample N    │                                    │
│  │ R_ψ(ẑ) │    │  times, take │                                    │
│  └───┬────┘    │  variance)   │                                    │
│      │         └──────┬───────┘                                    │
│      │                │                                            │
│      ▼                ▼                                            │
│  ┌─────────────────────────────────┐                               │
│  │   Expected Free Energy G(π)     │                               │
│  │ = -E[reward] - E[H(pred dist)]  │                               │
│  │                                 │                               │
│  │  → Update policy π_ξ            │                               │
│  │  → Update critic V_ω            │                               │
│  └─────────────────────────────────┘                               │
│                                                                    │
│  Repeat for H steps → imagined mutation trajectory                 │
│  z_t → ẑ_{t+1} → ẑ_{t+2} → ... → ẑ_{t+H}                           │
│                                                                    │
│  ★ No decoder needed — all planning happens in latent space ★      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Diffusion Process in Latent Space

**Forward process** (noise schedule): Given the target latent $\bar{z}_{t+1} \in \mathbb{R}^d$ from the EMA encoder, add Gaussian noise at level $\tau \in \{1, \ldots, K\}$:

$$
\bar{z}_{t+1}^{\tau} = \sqrt{\bar{\alpha}_\tau}\, \bar{z}_{t+1} + \sqrt{1 - \bar{\alpha}_\tau}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})
$$

where $\bar{\alpha}_\tau = \prod_{i=1}^{\tau} \alpha_i$ is the cumulative noise schedule (cosine schedule, $K = 20$ steps).

**Denoiser network** $D_\theta$: A small Transformer (4 layers, 8 heads, $d_{\text{model}} = 512$) that takes the noised latent, noise level, context latent, and action embedding:

$$
\hat{z}_{t+1}^0 = D_\theta\big(\bar{z}_{t+1}^{\tau},\; \tau,\; z_t,\; a_t^{\text{emb}}\big)
$$

**Conditioning mechanism:** $z_t$ and $a_t^{\text{emb}}$ are concatenated and injected via cross-attention:

- Query: $\bar{z}_{t+1}^{\tau}$ (the noised target)
- Key/Value: $[z_t \| a_t^{\text{emb}}]$ (context + action)
- Noise level $\tau$ injected via sinusoidal embedding added to each layer.

**Reverse process (sampling):** During imagination, generate $\hat{z}_{t+1}$ by iterative denoising:

$$
\hat{z}_{t+1}^{\tau-1} = \frac{1}{\sqrt{\alpha_\tau}}\Big(\hat{z}_{t+1}^{\tau} - \frac{1 - \alpha_\tau}{\sqrt{1 - \bar{\alpha}_\tau}} \cdot \epsilon_\theta(\hat{z}_{t+1}^{\tau}, \tau, z_t, a_t)\Big) + \sigma_\tau \mathbf{w}
$$

where $\mathbf{w} \sim \mathcal{N}(0, \mathbf{I})$ and $\sigma_\tau$ is the noise schedule variance. Starting from $\hat{z}_{t+1}^K \sim \mathcal{N}(0, \mathbf{I})$, iterate $K$ steps to get $\hat{z}_{t+1}^0$.

### 2.4 Loss Functions

The total training loss for Architecture A:

$$
\boxed{\mathcal{L}_A = \beta_{\text{diff}} \cdot \mathcal{L}_{\text{diff}} + \lambda \cdot \text{SIGReg}(Z) + \beta_{\text{rew}} \cdot \mathcal{L}_{\text{reward}}}
$$

**1. Denoising score-matching loss** $\mathcal{L}_{\text{diff}}$ (trains the diffusion predictor):

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{\tau \sim \mathcal{U}(1,K),\; \varepsilon \sim \mathcal{N}(0,\mathbf{I})}\Big[\big\|D_\theta(\bar{z}_{t+1}^{\tau}, \tau, z_t, a_t^{\text{emb}}) - \bar{z}_{t+1}\big\|_2^2\Big]
$$

This is the $x_0$-prediction formulation (predicting the clean latent directly). Equivalent to noise-prediction $\epsilon$-formulation up to reparametrisation.

**2. SIGReg** (prevents encoder collapse — applied on context encoder output $z_t$):

SIGReg enforces $z_t \sim \mathcal{N}(0, \mathbf{I})$ via random projections + Epps-Pulley normality testing + Cramér-Wold aggregation (see Section 1.2). A single hyperparameter $\lambda \in [0.01, 0.2]$ replaces VICReg's 6-7 competing loss terms.

**3. Reward loss** $\mathcal{L}_{\text{reward}}$ (trains the fitness predictor):

$$
\mathcal{L}_{\text{reward}} = \frac{1}{B}\sum_{i=1}^{B} \sum_{k} w_k \cdot \text{SmoothL1}\big(R_\psi(z_t^{(i)})_k,\; y_k^{(i)}\big)
$$

**Loss weights (defaults):** $\beta_{\text{diff}} = 1.0$, $\lambda = 0.1$ (SIGReg), $\beta_{\text{rew}} = 1.0$.

### 2.5 Uncertainty Quantification (Built-In)

The diffusion predictor provides **free uncertainty estimates** by sampling $N$ times from $p_\theta(z_{t+1} | z_t, a_t)$:

$$
\{\hat{z}_{t+1}^{(n)}\}_{n=1}^N \sim p_\theta(z_{t+1} | z_t, a_t), \quad N = 16
$$

**Epistemic + aleatoric uncertainty:**

$$
\text{Uncertainty}(z_t, a_t) = \frac{1}{N}\sum_{n=1}^N \|\hat{z}_{t+1}^{(n)} - \bar{\hat{z}}_{t+1}\|_2^2, \quad \bar{\hat{z}}_{t+1} = \frac{1}{N}\sum_{n=1}^N \hat{z}_{t+1}^{(n)}
$$

This directly feeds the **epistemic value** term in Expected Free Energy:

$$
G_{\text{epistemic}}(\pi) = -\mathbb{E}_{q}\Big[\mathbb{H}\big[p_\theta(z_{t+1} | z_t, a_t)\big]\Big] \approx -\frac{d}{2}\log\Big(2\pi e \cdot \text{Uncertainty}(z_t, a_t) / d\Big)
$$

High uncertainty → high epistemic value → policy explores → reduces model uncertainty → Active Inference.

### 2.6 Imagination-Based Policy Training

**Algorithm: Latent Diffusion JEPA Policy Optimisation**

```
Input: Replay buffer D of (s_t, a_t, r_t, s_{t+1}) transitions from DMS data
Output: Trained world model (f_θ, D_θ, R_ψ) and policy π_ξ

1. WORLD MODEL TRAINING (every step):
   Sample batch {(s_t, a_t, r_t, s_{t+1})} from D
   Compute z_t = f_θ(s_t), z̄_{t+1} = f_θ̄(s_{t+1})  [stop-grad on z̄]
   Compute a_t^emb = e_α(a_t)
   Sample τ ~ U(1,K), ε ~ N(0,I)
   Compute z̄^τ_{t+1} = √ᾱ_τ · z̄_{t+1} + √(1-ᾱ_τ) · ε
   Predict ẑ^0_{t+1} = D_θ(z̄^τ_{t+1}, τ, z_t, a_t^emb)
   Compute L_A = β_diff·||ẑ^0_{t+1} - z̄_{t+1}||² + λ·SIGReg(Z) + β_rew·L_reward
   Update θ, α, ψ by ∇L_A
   Update θ̄ ← τ·θ̄ + (1-τ)·θ  [EMA]

2. IMAGINATION ROLLOUT (every N_imagine steps):
   Sample initial z_0 from encoded replay buffer states
   For t = 0, ..., H-1:
     a_t ~ π_ξ(·|z_t)                           [sample action from policy]
     Sample N latents: {ẑ^(n)_{t+1}} ~ p_θ(·|z_t, a_t)  [N diffusion samples]
     ẑ_{t+1} = mean({ẑ^(n)_{t+1}})              [use mean for forward pass]
     r̂_t = R_ψ(ẑ_{t+1})                         [predict fitness]
     u_t = Var({ẑ^(n)_{t+1}})                    [uncertainty from samples]
     g_t = -r̂_t - η·u_t                          [Expected Free Energy]
     z_{t+1} ← ẑ_{t+1}

3. POLICY UPDATE (on imagined trajectories):
   Compute λ-returns V^λ_t from {g_t, ..., g_{t+H}}
   Update critic: ∇_ω ||V_ω(z_t) - V^λ_t||²
   Update actor:  ∇_ξ E[log π_ξ(a_t|z_t) · (-g_t - V_ω(z_t))]
```

### 2.7 Alternatives Within Architecture A

If using the Latent Diffusion JEPA world model, the following components can be swapped:

| Component                       | Default              | Alternative 1           | Alternative 2         |
| ------------------------------- | -------------------- | ----------------------- | --------------------- |
| **Sequence encoder**      | ESM-2 650M (frozen)  | ESM-2 650M (fine-tuned) | ProGen2-medium        |
| **Structure encoder**     | GVP-GNN (3 layers)   | EGNN                    | PaiNN                 |
| **Diffusion schedule**    | Cosine ($K=20$)    | Linear ($K=50$)       | Flow matching (ODE)   |
| **Denoiser architecture** | Transformer (4L, 8H) | U-Net 1D                | MLP + residual        |
| **Policy**                | SAC (actor-critic)   | PPO                     | MCTS                  |
| **Reward head**           | Multi-task MLP       | S3F multi-scale         | Tranception zero-shot |
| **Fitness data**          | ProteinGym DMS       | Tsuboyama ΔΔG         | Custom assay          |

---

## 3. Architecture B — Energy-Based JEPA (Non-Generative)

### 3.1 Overview

The world model predictor is a **deterministic Transformer** that maps $(z_t, a_t) \to \hat{z}_{t+1}$ in a single forward pass, trained with a prediction loss and SIGReg on the encoder output to prevent representational collapse. No sampling, no decoder — fastest inference. Uncertainty is estimated externally via an ensemble of predictors or an evidential deep learning head.

### 3.2 Architecture Diagram

```
Architecture B — Energy-Based JEPA for ProteinDreamer
═══════════════════════════════════════════════════════

                        TRAINING PHASE
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Protein State s_t          Protein State s_{t+1}                  │
│  (seq + struct)             (seq + struct after mutation)          │
│       │                            │                               │
│       ▼                            ▼                               │
│  ┌──────────┐              ┌──────────────┐                        │
│  │ Context  │              │   Target     │                        │
│  │ Encoder  │              │  Encoder     │                        │
│  │  f_θ     │              │   f_θ̄ (EMA)  │                        │
│  └────┬─────┘              └──────┬───────┘                        │
│       │                           │                                │
│       ▼                           ▼                                │
│      z_t                        z̄_{t+1}  ◄── target (stop-grad)    │
│       │                           │                                │
│       │    ┌───────────┐          │                                │
│       │    │  Action   │          │                                │
│       │    │ Encoder   │          │                                │
│       │    │  e_α(a_t) │          │                                │
│       │    └─────┬─────┘          │                                │
│       │          │                │                                │
│       ▼          ▼                │                                │
│  ┌──────────────────────┐         │                                │
│  │  Deterministic       │         │                                │
│  │  JEPA Predictor g_φ  │         │                                │
│  │  (Transformer 4L 8H) │         │                                │
│  │                      |         │                                │
│  │  ẑ_{t+1} = g_φ(z_t,  │         │                                │
│  │             a_t^emb) │         │                                │
│  └──────────┬───────────┘         │                                │
│             │                     │                                │
│             ▼                     ▼                                │
│       ┌────────────────────────────────┐                           │
│       │  JEPA Energy + SIGReg Loss    │                           │
│       │                                │                           │
│       │  E(ẑ, z̄) = ||ẑ - z̄||²         │                           │
│       │  + λ·SIGReg(Z)               │                           │
│       │  (enforce z_t ~ N(0,I))      │                           │
│       └─────────────┴──────────────────┘                           │
│                     │                                              │
│                     ▼                                              │
│  Combined Loss:                                                    │
│  L = β_jepa · L_pred + λ · SIGReg(Z) + β_rew · L_reward             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

                      IMAGINATION / PLANNING PHASE
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Current latent z_t                                                │
│       │                                                            │
│       │    Policy π_ξ(a|z_t) proposes mutation a_t                 │
│       │          │                                                 │
│       ▼          ▼                                                 │
│  ┌─────────────────────────────────┐                               │
│  │  Deterministic JEPA Predictor   │                               │
│  │  g_φ(z_t, a_t^emb)              │                               │
│  │  → ẑ_{t+1}                      │  ◄── single forward pass      │
│  └──────────┬──────────────────────┘                               │
│             │                                                      │
│     ┌───────┼──────────────────┐                                   │
│     │       │                  │                                   │
│     ▼       │                  ▼                                   │
│  ┌────────┐ │    ┌──────────────────────┐                          │
│  │ Reward │ │    │ Uncertainty Ensemble │                          │
│  │  Head  │ │    │                      │                          │
│  │ R_ψ(ẑ)││ |    │ M predictors g_φ^(m) │                          │
│  └───┬────┘ │    │ u_t = Var({ẑ^(m)})   │                          │
│      │      │    └──────────┬───────────┘                          │
│      │      │               │                                      │
│      ▼      │               ▼                                      │
│  ┌─────────────────────────────────┐                               │
│  │   Expected Free Energy G(π)     │                               │
│  │ = -E[reward] - η·uncertainty    │                               │
│  │                                 │                               │
│  │  → Update policy π_ξ            │                               │
│  │  → Update critic V_ω            │                               │
│  └─────────────────────────────────┘                               │
│                                                                    │
│  Repeat for H steps → imagined mutation trajectory                 │
│  z_t → ẑ_{t+1} → ẑ_{t+2} → ... → ẑ_{t+H}                           │
│                                                                    │
│  ★ No decoder, no sampling — fastest imagination ★                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 Deterministic JEPA Predictor $g_\phi$

A Transformer-based predictor that maps $(z_t, a_t^{\text{emb}}) \to \hat{z}_{t+1}$:

$$
\hat{z}_{t+1} = g_\phi(z_t, a_t^{\text{emb}}) = \text{TransformerBlock}^{(4)}\big([z_t \| a_t^{\text{emb}}]\big)[:d]
$$

**Architecture details:**

- Input: concatenation $[z_t \| a_t^{\text{emb}}] \in \mathbb{R}^{d + d_a}$
- 4 Transformer layers, 8 attention heads, $d_{\text{model}} = 512$
- Output projection: linear layer → $\hat{z}_{t+1} \in \mathbb{R}^d$
- Layer normalisation + GELU activations
- Residual connections

The predictor is **deterministic** — the same input always produces the same output. This is a single forward pass.

### 3.4 Loss Functions

The total training loss for Architecture B:

$$
\boxed{\mathcal{L}_B = \beta_{\text{jepa}} \cdot \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z) + \beta_{\text{rew}} \cdot \mathcal{L}_{\text{reward}}}
$$

**1. Prediction loss** $\mathcal{L}_{\text{pred}}$ (trains the predictor to match target encoder output):

$$
\mathcal{L}_{\text{pred}} = \frac{1}{B}\sum_{i=1}^{B} \big\|g_\phi(z_t^{(i)}, a_t^{(i)\text{emb}}) - \text{sg}(\bar{z}_{t+1}^{(i)})\big\|_2^2
$$

where $\text{sg}(\cdot)$ denotes stop-gradient (the target encoder is not updated by this loss).

**2. SIGReg** (prevents encoder collapse — applied on context encoder output $z_t$):

Identical to Architecture A (Section 2.4). SIGReg enforces $z_t \sim \mathcal{N}(0, \mathbf{I})$ via random projections + Epps-Pulley normality testing + Cramér-Wold aggregation. Single hyperparameter $\lambda \in [0.01, 0.2]$.

**3. Reward loss** $\mathcal{L}_{\text{reward}}$ — identical to Architecture A (Section 1.5).

**Loss weights (defaults):** $\beta_{\text{jepa}} = 1.0$, $\lambda = 0.1$ (SIGReg), $\beta_{\text{rew}} = 1.0$.

### 3.5 Uncertainty Quantification (External)

Since the deterministic predictor produces a single prediction (no distributional samples), uncertainty must be estimated externally. Two options:

**Option 1 — Predictor Ensemble ($M = 5$):**

Train $M$ independent JEPA predictors $\{g_{\phi_1}, \ldots, g_{\phi_M}\}$ with different random initialisations. Uncertainty at $(z_t, a_t)$:

$$
\text{Uncertainty}_{\text{ens}}(z_t, a_t) = \frac{1}{M}\sum_{m=1}^M \big\|g_{\phi_m}(z_t, a_t^{\text{emb}}) - \bar{g}(z_t, a_t^{\text{emb}})\big\|_2^2
$$

$$
\bar{g}(z_t, a_t^{\text{emb}}) = \frac{1}{M}\sum_{m=1}^M g_{\phi_m}(z_t, a_t^{\text{emb}})
$$

$M \times$ compute cost during imagination, but each forward pass is still single-step (no denoising iterations).

**Option 2 — Evidential Deep Learning:**

Replace the point-estimate predictor output with a Normal-Inverse-Gamma (NIG) distribution:

$$
g_\phi(z_t, a_t^{\text{emb}}) \to (\hat{\mu}, \hat{\nu}, \hat{\alpha}, \hat{\beta})
$$

where $\hat{\mu}$ is the predicted mean (= $\hat{z}_{t+1}$), and $(\hat{\nu}, \hat{\alpha}, \hat{\beta})$ parameterise the uncertainty:

- **Aleatoric uncertainty:** $\mathbb{E}[\sigma^2] = \hat{\beta} / (\hat{\alpha} - 1)$
- **Epistemic uncertainty:** $\text{Var}[\mu] = \hat{\beta} / (\hat{\nu}(\hat{\alpha} - 1))$

Trained with the evidential loss:

$$
\mathcal{L}_{\text{evid}} = \frac{1}{2}\log\frac{\pi}{\hat{\nu}} - \hat{\alpha}\log\Omega + \Big(\hat{\alpha} + \frac{1}{2}\Big)\log\big((\hat{z}_{t+1} - \hat{\mu})^2 \hat{\nu} + \Omega\big) + \log\frac{\Gamma(\hat{\alpha})}{\Gamma(\hat{\alpha} + \frac{1}{2})}
$$

where $\Omega = 2\hat{\beta}(1 + \hat{\nu})$. This provides uncertainty in a **single forward pass** — no ensemble needed.

### 3.6 Imagination-Based Policy Training

**Algorithm: Energy-Based JEPA Policy Optimisation**

```
Input: Replay buffer D of (s_t, a_t, r_t, s_{t+1}) transitions from DMS data
Output: Trained world model (f_θ, g_φ, R_ψ) and policy π_ξ

1. WORLD MODEL TRAINING (every step):
   Sample batch {(s_t, a_t, r_t, s_{t+1})} from D
   Compute z_t = f_θ(s_t), z̄_{t+1} = f_θ̄(s_{t+1})  [stop-grad on z̄]
   Compute a_t^emb = e_α(a_t)
   Predict ẑ_{t+1} = g_φ(z_t, a_t^emb)
   Compute L_B = β_jepa·L_pred(ẑ_{t+1}, z̄_{t+1}) + λ·SIGReg(Z) + β_rew·L_reward
   Update θ, φ, α, ψ by ∇L_B
   Update θ̄ ← τ·θ̄ + (1-τ)·θ  [EMA]

2. IMAGINATION ROLLOUT (every N_imagine steps):
   Sample initial z_0 from encoded replay buffer states
   For t = 0, ..., H-1:
     a_t ~ π_ξ(·|z_t)                             [sample action from policy]
     ẑ_{t+1} = g_φ(z_t, a_t^emb)                  [single forward pass]
     r̂_t = R_ψ(ẑ_{t+1})                           [predict fitness]
     u_t = Uncertainty(z_t, a_t)                    [ensemble or evidential]
     g_t = -r̂_t - η·u_t                            [Expected Free Energy]
     z_{t+1} ← ẑ_{t+1}

3. POLICY UPDATE (on imagined trajectories):
   Compute λ-returns V^λ_t from {g_t, ..., g_{t+H}}
   Update critic: ∇_ω ||V_ω(z_t) - V^λ_t||²
   Update actor:  ∇_ξ E[log π_ξ(a_t|z_t) · (-g_t - V_ω(z_t))]
```

### 3.7 Alternatives Within Architecture B

If using the Energy-Based JEPA world model, the following components can be swapped:

| Component                     | Default              | Alternative 1           | Alternative 2          |
| ----------------------------- | -------------------- | ----------------------- | ---------------------- |
| **Sequence encoder**    | ESM-2 650M (frozen)  | ESM-2 650M (fine-tuned) | ProGen2-medium         |
| **Structure encoder**   | GVP-GNN (3 layers)   | EGNN                    | None (sequence-only)   |
| **Predictor**           | Transformer (4L, 8H) | MLP (3×512)            | GRU + MLP              |
| **Collapse prevention** | SIGReg               | Barlow Twins            | VICReg (legacy)        |
| **Uncertainty**         | Ensemble ($M=5$)   | Evidential DL           | MC Dropout             |
| **Policy**              | SAC (actor-critic)   | PPO                     | MCTS                   |
| **Reward head**         | Multi-task MLP       | S3F multi-scale         | ESM-2 masked marginals |

---

## 4. Comparison Summary

### 4.1 Architecture Comparison Table

| Property                       | Architecture A (Latent Diffusion JEPA)                                                    | Architecture B (Energy-Based JEPA)                          |
| ------------------------------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Predictor type**       | Conditional diffusion in latent space                                                     | Deterministic Transformer/MLP                               |
| **Output**               | Distribution$p_\theta(z_{t+1} \| z_t, a_t)$                                             | Point estimate$\hat{z}_{t+1}$                             |
| **Multi-modal**          | Yes — captures multiple outcomes                                                         | No — averages over modes                                   |
| **Uncertainty**          | Built-in (sample variance)                                                                | External (ensemble / evidential)                            |
| **Inference cost**       | $K$ denoising steps ($K \approx 20$) per prediction                                   | 1 forward pass per prediction                               |
| **Training loss**        | $\mathcal{L}_{\text{diff}} + \lambda \cdot \text{SIGReg}(Z) + \mathcal{L}_{\text{reward}}$ | $\mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z) + \mathcal{L}_{\text{reward}}$ |
| **Decoder needed**       | No                                                                                        | No                                                          |
| **Active Inference fit** | Natural — diffusion entropy feeds EFE                                                    | Requires external uncertainty                               |
| **Best for**             | Rugged/multi-modal landscapes, high-accuracy design                                       | Rapid screening, large-scale search                         |
| **Novelty**              | High — latent diffusion + JEPA is unexplored                                             | Moderate — direct Causal-JEPA transfer                     |

### 4.2 When to Use Which

- **Architecture A** when:

  - The fitness landscape is rugged with multiple basins (multi-modal transitions after mutation)
  - Accurate multi-step rollouts are needed (diffusion avoids error compounding better than deterministic predictors)
  - Uncertainty-driven exploration is critical (e.g., early-stage optimisation with little data)
  - Computational budget allows $K \times$ slowdown per prediction step
- **Architecture B** when:

  - Speed is the priority — screening thousands of candidate mutation paths
  - The fitness landscape is relatively smooth (single-mode transitions)
  - The uncertainty module (ensemble/evidential) provides sufficient exploration signal
  - As a warm-start: train Architecture B first (faster), then upgrade to Architecture A for refinement

### 4.3 Shared Strengths

Both architectures share:

- **No decoder during planning** — all imagination happens in latent space, no expensive structure reconstruction
- **Active Inference policy** — principled exploration-exploitation via Expected Free Energy
- **Pre-trained protein representations** — ESM-2 + GVP-GNN provide strong inductive bias
- **JEPA training paradigm** — EMA target encoder prevents collapse, no reconstruction loss needed
- **Modular design** — any component (encoder, predictor, policy, reward) can be swapped independently

### 4.4 Full System Diagram — Both Architectures

```
                    ProteinDreamer — Complete System
════════════════════════════════════════════════════════════

  Wild-type protein                    Target fitness
  (sequence + PDB)                     (stability, binding, ...)
        │                                      │
        ▼                                      │
  ┌──────────────┐                             │
  │  ESMFold /   │  (cheap structure oracle)   │
  │  AlphaFold2  │                             │
  └──────┬───────┘                             │
         │                                     │
         ▼                                     │
  ┌─────────────────────────────────────────────────────────────┐
  │                    WORLD MODEL                              │
  │                                                             │
  │  ┌────────────────┐                ┌───────────────────┐    │
  │  │ Context Encoder│                │  Target Encoder   │    │
  │  │  ESM-2 + GVP   │                │  (EMA copy)       │    │
  │  │  f_θ(s_t)→z_t  │                │  f_θ̄(s_{t+1})     │    │
  │  └───────┬────────┘                │  →z̄_{t+1}         │    │
  │          │                         └────────┬──────────┘    │
  │          │    ┌──────────────┐              │               │
  │          │    │Action Encoder│              │               │
  │          │    │e_α(mutation) │              │               │
  │          │    └──────┬───────┘              │               │
  │          │           │                      │               │
  │          ▼           ▼                      ▼               │
  │    ┌─────────────────────────────────────────────────┐      │
  │    │              JEPA Predictor                     │      │
  │    │                                                 │      │
  │    │  Choose one:                                    │      │
  │    │  ┌──────────────────┐  ┌──────────────────────┐ │      │
  │    │  │ Arch A: Latent   │  │ Arch B: Deterministic│ │      │
  │    │  │ Diffusion (K=20) │  │ MLP/Transformer      │ │      │
  │    │  │ → samples from   │  │ → single ẑ_{t+1}     │ │      │
  │    │  │   p(z_{t+1}|     │  │   = g_φ(z_t, a_t)    │ │      │
  │    │  │     z_t, a_t)    │  │                      │ │      │
  │    │  └──────────────────┘  └──────────────────────┘ │      │
  │    └─────────────────────┬───────────────────────────┘      │
  │                          │                                  │
  │              ┌───────────┼───────────┐                      │
  │              │           │           │                      │
  │              ▼           ▼           ▼                      │
  │       ┌──────────┐ ┌──────────┐ ┌──────────────┐            │
  │       │  Reward  │ │Uncertain-│ │  Continue    │            │
  │       │  Head    │ │ty Module | │  Predictor   │            │
  │       │  R_ψ(z)  │ │(built-in │ │  (optional)  │            │
  │       │  →fitness│ │ or ext.) │ │              │            │
  │       └────┬─────┘ └────┬─────┘ └──────────────┘            │
  │            │            │                                   │
  └────────────┼────────────┼───────────────────────────────────┘
               │            │
               ▼            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                 ACTIVE INFERENCE POLICY                      │
  │                                                              │
  │  Expected Free Energy:                                       │
  │  G(π) = -E[reward] - η·E[uncertainty]                        │
  │         ─────────── ──────────────────                       │
  │         exploit         explore                              │
  │                                                              │
  │  ┌────────────┐    ┌────────────┐                            │
  │  │   Actor    │    │   Critic   │                            │
  │  │ π_ξ(a|z)   │    │  V_ω(z)    │                            │
  │  │ →mutation  │    │  →value    │                            │
  │  └─────┬──────┘    └────────────┘                            │
  │        │                                                     │
  └────────┼─────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  Candidate Mutation Paths│
  │  (ranked by EFE)         │
  │                          │
  │  1. WT→M1→M2→M5 (best)   │
  │  2. WT→M3→M7→M2          │
  │  3. WT→M1→M4→M9          │
  │  ...                     │
  └──────────┬───────────────┘
             │
     ┌───────┴──────────┐
     │                  │
     ▼                  ▼
  ┌────────────┐  ┌──────────────┐
  │  Validate  │  │  Active      │
  │  (ESMFold  │  │  Learning    │
  │  / AF2)    │  │  (wet-lab    │
  │            │  │   loop)      │
  └────────────┘  └──────────────┘
```
