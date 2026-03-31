<p align="center">
  <img src="frontend/public/logo.svg" alt="BioDreamer Logo" width="300" />
</p>

<h1 align="center">BioDreamer: World Models for Biological Design</h1>

<p align="center">
  <a href="https://loqmansamani.github.io/bio_dreamer/"><img src="https://img.shields.io/badge/Website-BioDreamer-blue?style=flat&logo=github-pages" alt="Website" /></a>
  <a href="https://loqmansamani.github.io/bio_dreamer/blog/"><img src="https://img.shields.io/badge/Blog-Posts%20%26%20Articles-purple?style=flat&logo=readme" alt="Blog" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js" alt="Next.js" /></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-FFD21E" alt="Hugging Face" /></a>
  <a href="https://github.com/LoqmanSamani/bio_dreamer/actions"><img src="https://img.shields.io/github/actions/workflow/status/LoqmanSamani/bio_dreamer/deploy.yml?branch=systembiology&label=deploy&logo=github-actions" alt="GitHub Actions" /></a>
</p>

---

## Why BioDreamer?

Biological design at every scale, from molecular dynamics to protein engineering to cellular reprogramming, runs into the same bottleneck. Real experiments are slow and expensive. An MD simulation of one protein can burn days on a GPU cluster. A round of directed evolution takes months and thousands of dollars. A genome-wide CRISPR screen consumes millions of cells and weeks of bench time.

Most ML approaches to these problems are one-shot. They generate a candidate, evaluate it, and repeat without any internal model of how the system actually behaves. No existing framework does what a good experimentalist would. A good experimentalist mentally simulates outcomes, plans multi-step strategies, and updates their intuition after each round of feedback.

BioDreamer applies **world models** from model-based reinforcement learning to biology. The agent learns a latent-space simulator (a JEPA-based world model) of the biological environment and then plans optimal interventions *in imagination* before committing to expensive real-world queries. The planning is grounded in **Active Inference** and the Free Energy Principle, giving the agent a principled way to balance exploitation (pursuing high-fitness regions) with exploration (reducing model uncertainty).

---

## Three Modules, One Architecture

| Scale | Module | What It Does | Actions | Reward |
|---|---|---|---|---|
| **Atomic** | **MolDreamer** | Learns molecular dynamics in latent space | Force/parameter changes, mutations | Binding ΔG, stability, SASA |
| **Protein** | **ProteinDreamer** | Navigates protein fitness landscapes via dreaming | Sequence mutations / edits | ΔΔG, Kd, kcat, expression |
| **Cellular** | **CellDreamer** | Plans cell reprogramming perturbation strategies | Gene knockouts, drug treatments | Distance to target cell state |

All three modules share a common backbone built on the **Joint-Embedding Predictive Architecture (JEPA)**.

```
Observation → Encoder → Latent State (z_t)
                            ↓
                    JEPA Predictor (z_t, action → ẑ_{t+1})
                            ↓
                    Reward Head (z_t → fitness)
                            ↓
                    Active Inference Policy (plans in imagination)
```

The key insight is that JEPA operates entirely in latent space. There is no decoder in the planning loop. The predictor maps the current latent state and a proposed action to a predicted next-state embedding, and the reward head scores that embedding directly. This makes imagination rollouts fast, because the agent never reconstructs full observations during planning.

The library supports two primary JEPA backends. **Latent Diffusion JEPA** uses a conditional denoising diffusion model as the predictor, capturing the multi-modal stochastic nature of biological transitions (a single mutation can lead to distinct structural outcomes). The diffusion sample variance provides built-in uncertainty estimates that feed directly into the Active Inference exploration term. **Energy-Based JEPA** uses a deterministic Transformer predictor with SIGReg regularization, trading distributional expressiveness for inference speed. Both backends share the same encoder, reward head, and policy interface, so they can be swapped without changing the rest of the pipeline.

The policy selects actions by minimising **expected free energy**, which naturally decomposes into pragmatic value (seek high fitness) and epistemic value (seek states where the model is uncertain). This is a direct application of Active Inference (Friston, 2010) to biological design, and the mathematical connection between JEPA energy and variational free energy makes the framework theoretically coherent rather than ad hoc.

---

## Project Structure

```
bio_dreamer/
│
├── biodreamer/                          # Core Python library (PyTorch)
│   ├── __init__.py
│   ├── core/                            # Shared base classes
│   │   ├── world_model.py               #   WorldModel (encoder + JEPA predictor + reward)
│   │   ├── encoder.py                   #   BaseEncoder interface
│   │   ├── dynamics.py                  #   BaseDynamics (JEPA predictor backends)
│   │   ├── decoder.py                   #   BaseDecoder (optional, for evaluation only)
│   │   ├── reward.py                    #   BaseRewardHead (multi-objective)
│   │   ├── policy.py                    #   BasePolicy (PPO, SAC, MCTS, GFlowNet)
│   │   └── active_inference.py          #   Active Inference / Expected Free Energy
│   │
│   ├── protein_dreamer/                 # Protein fitness landscape module
│   │   ├── encoder.py                   #   ESM-2 (sequence) + GVP-GNN (structure)
│   │   ├── dynamics.py                  #   JEPA predictor (diffusion or energy-based)
│   │   ├── decoder.py                   #   Sequence + structure + pLDDT (eval only)
│   │   ├── reward.py                    #   ΔΔG, Kd, kcat multi-objective
│   │   ├── policy.py                    #   ProteinPPO / SAC / MCTS
│   │   ├── environment.py               #   DMS lookup, ESMFold, wet-lab oracles
│   │   ├── uncertainty.py               #   Ensemble, evidential, MC-dropout
│   │   └── data/                        #   ProteinGym, Tsuboyama datasets
│   │
│   ├── mol_dreamer/                     # Molecular dynamics module
│   │   ├── encoder.py                   #   SE(3)-equivariant GNN (EGNN/PaiNN)
│   │   ├── dynamics.py                  #   Latent diffusion / neural SDE
│   │   ├── decoder.py                   #   Coordinate + contact map reconstruction
│   │   ├── reward.py                    #   ΔG, Tm, SASA property predictors
│   │   ├── policy.py                    #   Continuous-action SAC
│   │   ├── environment.py               #   Gym-like MD env (OpenMM oracle)
│   │   └── data/                        #   ATLAS, D.E. Shaw trajectories
│   │
│   ├── cell_dreamer/                    # Cellular reprogramming module
│   │   ├── encoder.py                   #   scVI-style VAE on scRNA-seq
│   │   ├── dynamics.py                  #   Neural ODE/SDE, perturbation-conditioned
│   │   ├── decoder.py                   #   ZINB gene expression reconstruction
│   │   ├── reward.py                    #   Cell state distance (cosine, Wasserstein)
│   │   ├── policy.py                    #   Combinatorial perturbation planner
│   │   ├── environment.py               #   Perturb-seq replay, SERGIO simulator
│   │   └── data/                        #   Norman 2019, SciPlex datasets
│   │
│   ├── hub/                             # Hugging Face Hub integration
│   │   ├── registry.py                  #   Model catalogue (ESM-2, Geneformer, etc.)
│   │   ├── upload.py                    #   Push models to HF Hub (safetensors)
│   │   ├── download.py                  #   Pull models with caching
│   │   └── model_card.py                #   Auto-generate HF model cards
│   │
│   ├── training/                        # Training infrastructure
│   │   ├── trainer.py                   #   BaseTrainer (DDP, mixed precision, wandb)
│   │   ├── world_model_trainer.py       #   JEPA world model training loop
│   │   ├── policy_trainer.py            #   Imagination-based actor-critic training
│   │   ├── active_learning.py           #   Dream → Propose → Evaluate → Update loop
│   │   ├── callbacks.py                 #   Wandb, checkpointing, early stopping
│   │   └── schedulers.py               #   WarmupCosine, CyclicKL, KLBalance
│   │
│   ├── inference/                       # Inference engine
│   │   ├── pipeline.py                  #   End-to-end inference pipeline
│   │   ├── dreamer.py                   #   Imagination rollout engine
│   │   └── candidate_ranker.py          #   Fitness + diversity + Pareto ranking
│   │
│   └── utils/                           # Shared utilities
│       ├── logging.py                   #   Structured logging (console + wandb)
│       ├── metrics.py                   #   Spearman ρ, NDCG, calibration error
│       ├── visualization.py             #   Plots + Mol* data + Plotly
│       └── io.py                        #   YAML, PDB, FASTA, CSV I/O helpers
│
├── server/                              # FastAPI backend
│   ├── main.py                          #   App factory, router registration, CORS
│   ├── config.py                        #   Pydantic settings (env vars)
│   ├── routers/                         #   REST endpoints per module + jobs + models
│   ├── schemas/                         #   Pydantic request/response models
│   ├── services/                        #   Business logic per module
│   ├── middleware/                       #   Auth (API key), rate limiting
│   └── workers/                         #   Redis-based GPU job runner
│
├── frontend/                            # Next.js 14 + React 18 + TypeScript
│   ├── src/app/                         #   App Router pages (dashboard, modules, jobs, blog)
│   ├── src/components/                  #   UI components
│   │   ├── layout/                      #     Navbar, Sidebar, Footer
│   │   ├── common/                      #     MolViewer, ProteinViewer, FitnessPlot, ...
│   │   ├── protein-dreamer/             #     ProteinDreamerForm, ProteinDreamerResults
│   │   ├── mol-dreamer/                 #     MolDreamerForm, MolDreamerResults
│   │   └── cell-dreamer/                #     CellDreamerForm, CellDreamerResults
│   ├── src/components/blog/              #     BlogPostCard, BlogRenderer
│   ├── src/hooks/                       #   useApi, useJob, useModel
│   ├── src/lib/                         #   API client, TypeScript types
│   └── src/styles/                      #   Tailwind globals
│
├── blog/                                # Blog (Markdown + standalone HTML articles)
│   ├── README.md                        #   Blog index and writing guide
│   ├── posts/                           #   Markdown posts with YAML frontmatter
│   └── assets/                          #   Images and media for posts
│
├── configs/                             # YAML configs per module + server
├── docs/                                # Architecture, API reference, tutorials, model cards
├── tests/                               # Unit, integration, e2e tests
├── scripts/                             # CLI for download, train, evaluate, export_to_hub
├── notebooks/                           # Jupyter notebooks for exploration and training
├── docker/                              # Dockerfiles for backend, frontend, GPU worker
│
├── pyproject.toml                       # Python project config + dependencies
├── docker-compose.yml                   # Full stack (backend + frontend + worker + Redis)
├── Makefile                             # Dev commands (install, train, test, docker-up)
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML framework | PyTorch 2.x |
| Protein language models | ESM-2 via Hugging Face `transformers` |
| Graph neural networks | PyTorch Geometric (`torch-geometric`) |
| Neural ODEs/SDEs | `torchdiffeq` |
| scRNA-seq processing | `scanpy`, `anndata` |
| Model hosting | Hugging Face Hub (`huggingface_hub`) |
| Experiment tracking | Weights & Biases (`wandb`) |
| Backend API | FastAPI + Uvicorn |
| Job queue | Redis |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| 3D molecular viewer | Mol* (Molstar) |
| Containerisation | Docker + docker-compose |

---

## Dependencies

### Python (ML + Backend)

```
torch >= 2.1
torch-geometric >= 2.4
torchdiffeq >= 0.2
transformers >= 4.35
huggingface-hub >= 0.19
safetensors >= 0.4
fastapi >= 0.104
uvicorn >= 0.24
redis >= 5.0
pydantic >= 2.5
pydantic-settings >= 2.1
wandb >= 0.16
scanpy >= 1.9
anndata >= 0.10
scipy >= 1.11
numpy >= 1.24
pandas >= 2.1
pyyaml >= 6.0
jinja2 >= 3.1
biopython >= 1.82
markdown-it-py >= 3.0
pygments >= 2.17
```

### Frontend (Node.js)

```
next >= 14.0
react >= 18.2
typescript >= 5.3
tailwindcss >= 3.4
molstar >= 4.0
recharts >= 2.10 (or plotly.js)
react-markdown >= 9.0
rehype-highlight >= 7.0
rehype-katex >= 7.0
remark-math >= 6.0
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/LoqmanSamani/bio_dreamer.git
cd bio_dreamer

# Install Python package (editable)
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend && npm install && cd ..

# Copy environment config
cp .env.example .env  # edit with your HF_TOKEN, etc.

# Run tests
make test

# Start full stack (Docker)
make docker-up

# Or start backend + frontend separately
make dev        # FastAPI on :8000
cd frontend && npm run dev  # Next.js on :3000
```

---

## Training

```bash
# Download datasets
bash scripts/download_data.sh --protein

# Train world model
python scripts/train_world_model.py \
    --module protein_dreamer \
    --config configs/protein_dreamer/default.yaml \
    --output-dir checkpoints/protein_dreamer/

# Train policy (imagination-based)
python scripts/train_policy.py \
    --module protein_dreamer \
    --world-model checkpoints/protein_dreamer/world_model.pt \
    --output-dir checkpoints/protein_dreamer/policy/

# Evaluate
python scripts/evaluate.py \
    --module protein_dreamer \
    --world-model checkpoints/protein_dreamer/world_model.pt \
    --dataset proteingym

# Export to Hugging Face Hub
python scripts/export_to_hub.py \
    --checkpoint checkpoints/protein_dreamer/world_model.pt \
    --repo-id username/biodreamer-protein-v1
```

---


## License

MIT
