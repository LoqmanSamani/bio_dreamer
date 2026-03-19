# Model Card: MolDreamer World Model

> Hugging Face-style model card for the MolDreamer latent world model.
>
> ## Sections to write
>
> - **Model Description:** SE(3)-equivariant GNN encoder + latent diffusion dynamics + coordinate decoder trained on MD trajectory data. Predicts molecular property changes (ΔG, Tm, SASA) and evolves molecular states in latent space.
> - **Intended Use:** In-silico molecular dynamics surrogate for model-based RL optimisation of molecular properties.
> - **Training Data:** ATLAS protein dynamics, DE Shaw trajectories, PDBBind, Martini 3 CG datasets.
> - **Architecture Details:** Encoder type, latent dimensionality, dynamics model specification, decoder structure.
> - **Evaluation Metrics:** Trajectory RMSD vs ground-truth MD, property prediction MAE, rollout stability.
> - **Limitations & Biases:** Distribution of training proteins, timescale coverage, force field assumptions.
> - **How to Use:** Code snippet for loading from Hugging Face Hub and running inference.
