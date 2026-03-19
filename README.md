# BioDreamer — World Models for Biological Design

> *Teaching machines to dream about biology so we don't have to wait for every experiment.*

---

## Why BioDreamer?

Biological design — engineering proteins, simulating molecular dynamics, reprogramming cells — faces a fundamental bottleneck: **real experiments are slow and expensive**. An MD simulation of one protein can take days on a GPU cluster. A single round of directed evolution costs months and thousands of dollars. A genome-wide CRISPR screen requires millions of cells and weeks of work.

Current ML approaches are mostly **one-shot**: generate a candidate, hope it works, repeat. No existing system does what a skilled engineer would — **mentally simulate outcomes before committing**, plan multi-step strategies, and learn from each round of feedback.

**BioDreamer** solves this by applying *world models* from model-based reinforcement learning to biology. Instead of querying the real environment (MD simulator, wet lab, CRISPR screen), the agent learns a latent-space simulator and plans optimal interventions *in imagination* — replacing brute-force experimentation with intelligent, amortised, in-silico reasoning.

---

## Three Modules, One Architecture

| Scale | Module | What It Does | Actions | Reward |
|---|---|---|---|---|
| **Atomic** | **MolDreamer** | Learns molecular dynamics in latent space | Force/parameter changes, mutations | Binding ΔG, stability, SASA |
| **Protein** | **ProteinDreamer** | Navigates protein fitness landscapes via dreaming | Sequence mutations / edits | ΔΔG, Kd, kcat, expression |
| **Cellular** | **CellDreamer** | Plans cell reprogramming perturbation strategies | Gene knockouts, drug treatments | Distance to target cell state |

All three modules share a common skeleton:

```
Observation → Encoder → Latent State (z_t)
                            ↓
                    Dynamics Model (z_t, action → z_{t+1})
                            ↓
                    Reward Model (z_t → fitness)
                            ↓
                    Policy (RL agent plans actions in imagination)
                            ↓
                    Decoder → Predicted Outcome
```

The policy *dreams* multi-step trajectories inside the world model before proposing candidates, guided by **Active Inference** (Free Energy Principle) for principled exploration–exploitation trade-off.

---

## Project Structure

```
bio_dreamer/
│
├── biodreamer/                          # Core Python library (PyTorch)
│   ├── __init__.py
│   ├── core/                            # Shared base classes
│   │   ├── world_model.py               #   WorldModel (encoder+dynamics+decoder+reward)
│   │   ├── encoder.py                   #   BaseEncoder interface
│   │   ├── dynamics.py                  #   BaseDynamics interface
│   │   ├── decoder.py                   #   BaseDecoder interface
│   │   ├── reward.py                    #   BaseRewardHead (multi-objective)
│   │   ├── policy.py                    #   BasePolicy (PPO, SAC, MCTS, GFlowNet)
│   │   └── active_inference.py          #   Active Inference / FEP engine
│   │
│   ├── protein_dreamer/                 # Protein fitness landscape module
│   │   ├── encoder.py                   #   ESM-2 (sequence) + GVP-GNN (structure)
│   │   ├── dynamics.py                  #   Mutation-conditioned transitions
│   │   ├── decoder.py                   #   Sequence + structure + pLDDT
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
│   │   ├── world_model_trainer.py       #   DreamerV3-style training loop
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
│   ├── src/app/                         #   App Router pages (dashboard, modules, jobs)
│   ├── src/components/                  #   UI components
│   │   ├── layout/                      #     Navbar, Sidebar, Footer
│   │   ├── common/                      #     MolViewer, ProteinViewer, FitnessPlot, ...
│   │   ├── protein-dreamer/             #     ProteinDreamerForm, ProteinDreamerResults
│   │   ├── mol-dreamer/                 #     MolDreamerForm, MolDreamerResults
│   │   └── cell-dreamer/                #     CellDreamerForm, CellDreamerResults
│   ├── src/hooks/                       #   useApi, useJob, useModel
│   ├── src/lib/                         #   API client, TypeScript types
│   └── src/styles/                      #   Tailwind globals
│
├── configs/                             # YAML configs per module + server
├── docs/                                # Architecture, API reference, tutorials, model cards
├── tests/                               # Unit, integration, e2e tests
├── scripts/                             # CLI: download_data, train, evaluate, export_to_hub
├── notebooks/                           # Jupyter: exploration, training, active learning, hub
├── docker/                              # Dockerfiles: backend, frontend, GPU worker
│
├── pyproject.toml                       # Python project config + dependencies
├── docker-compose.yml                   # Full stack: backend + frontend + worker + Redis
├── Makefile                             # Dev commands (install, train, test, docker-up)
├── .env.example                         # Environment variable template
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
```

### Frontend (Node.js)

```
next >= 14.0
react >= 18.2
typescript >= 5.3
tailwindcss >= 3.4
molstar >= 4.0
recharts >= 2.10 (or plotly.js)
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
