## 1. ProteinDreamer: System Architecture Overview

ProteinDreamer is a **Latent Diffusion Joint-Embedding Predictive Architecture (JEPA)** designed for protein design. It frames the design process as an **Active Inference** task within a Markov Decision Process (MDP), allowing a policy to "dream" and optimize mutation trajectories in a compressed latent space.

## 2. Encoders and Latent Representation

### 2.1 Context Encoder ($E_c$)

The Context Encoder maps raw protein observations into a stable latent state $z_t \in \mathbb{R}^d$ ($d = 256$–$512$). It acts as the primary "perceiver" of the current protein state.

* **ESM-2 (650M or other models with the same functionality):** Processes the sequence $\mathbf{x}_t$ to generate per-residue embeddings ($L \times 1280$). This captures evolutionary and chemical context.
* **GVP-GNN:** A trainable structural encoder that processes 3D coordinates $\mathbf{C}_t$ (predicted via ESMFold/AlphaFold2) into structural embeddings ($L \times 1280$).
  * **Inputs:** C$\alpha$ coordinates for vector/scalar geometry, **pLDDT** as a node-level confidence scalar (weighting structural reliability), and the amino acid identity.
* **Fusion Pipeline:**
  1. **MeanPool:** Reduces the $L$ dimension to produce global summaries $h_{\text{seq}}$ and $h_{\text{struct}}$.
  2. **Concatenate:** Merges sequence and structure into a single $2560$-D vector.
  3. **Fusion MLP:** Compresses the merged vector into the final latent $z_t$.
  4. **SIGReg:** Applied to $z_t$ to enforce an isotropic Gaussian distribution ($\mathcal{N}(0, \mathbf{I})$), preventing representational collapse and optimizing the manifold for diffusion.

### 2.2 Action Encoder ($E_a$)

The Action Encoder is a trainable MLP that maps discrete mutation events (position $i$, old AA, new AA) into a continuous action space ($128$-D). This allows the model to learn the chemical "distance" between different amino acid substitutions and enables end-to-end gradient flow.

### 2.3 Target Encoder ($E_t$)

This encoder is a structural "twin" of the Context Encoder used to provide stable targets for the World Model.

* **Input:** Processes the state $s_{t+1}$ **after** a mutation has occurred.
* **Update Mechanism:** It uses **stop-gradient** and **Exponential Moving Average (EMA)** to update its weights ($f_{\bar{\theta}}$), serving as a slowly-shifting "anchor" to prevent training divergence.

## 3. World Model Predictor ($P_\phi$)

The World Model Predictor is the "engine" of the system, simulating protein dynamics in the latent space via a **Conditional Denoising Diffusion Model**.

* **Denoising Network ($D_\theta$):** A Transformer-based architecture that predicts the "clean" latent state from a noised version, conditioned on the current state $z_t$ and action $a_t^{\text{emb}}$ via cross-attention.
* **Built-in Uncertainty:** By sampling $N$ different denoising paths, the model calculates the variance of the predicted latents. This variance serves as a direct proxy for **Epistemic Uncertainty**, identifying mutation spaces the model hasn't explored.
* **Latent Dynamics:** Because it operates entirely in the compressed latent space, it can simulate thousands of mutation trajectories per second without needing slow external folding tools.

## 4. Reward Head (Fitness Predictor) $R_\psi$

The Reward Head is a "digital assay" that translates the abstract latent state $z_t$ into biological utility.

* **Architecture:** A multi-task MLP ($3 \times 256$ layers) with a shared backbone and task-specific heads.
* **Output:** A **Multi-Objective** vector $\hat{r}_t = [\hat{y}^{\text{stab}}_t, \hat{y}^{\text{bind}}_t, \hat{y}^{\text{act}}_t]$, allowing the model to balance trade-offs like stability vs. binding affinity.
* **Training:** Supervised learning on DMS data (ProteinGym/Tsuboyama) using a **SmoothL1 loss** for robustness against experimental noise.

## 5. Active Inference Policy ($\pi_\xi$)

The Active Inference Policy is a stochastic neural network (Actor) that maps latent states to mutation probabilities. It optimizes itself through **Imagination Rollouts**.

* **The Planning Phase:** Every $N$ training steps, the Actor proposes mutations that the World Model "imagines" forward for $H$ steps. This multi-step dreaming allows the **Critic ($V_\omega$)** to estimate the long-term value of a mutation path.
* **Expected Free Energy ($G$):** The policy minimizes $G$, balancing two goals:
  1. **Exploitation (Pragmatic Value):** Maximizing fitness scores predicted by $R_\psi$.
  2. **Exploration (Epistemic Value):** Seeking out mutations with high uncertainty ($u_t$) to improve the model's knowledge of the fitness landscape.
* **Strategic Advantage:** No decoder is needed for design. All optimization occurs in the latent space, and only the final optimized sequences are reconstructed for synthesis.

## 6. Training Loss Summary

The total training objective ($\mathcal{L}_A$) balances representation, dynamics, and fitness:

$$
\mathcal{L}_A = \beta_{\text{diff}} \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{SIGReg}} + \beta_{\text{rew}} \mathcal{L}_{\text{reward}}
$$

1. **$\mathcal{L}_{\text{diff}}$:** Trains the diffusion predictor to accurately transition between states.
2. **$\mathcal{L}_{\text{SIGReg}}$:** Prevents latent collapse by enforcing a Gaussian manifold.
3. **$\mathcal{L}_{\text{reward}}$:** Grounds the latent space in experimental biological reality.
