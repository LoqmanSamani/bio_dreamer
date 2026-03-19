# Model Card: CellDreamer World Model

> Hugging Face-style model card for the CellDreamer gene-regulatory world model.
>
> ## Sections to write
>
> - **Model Description:** VAE encoder (scRNA-seq profiles → latent cell state) + Neural ODE/SDE dynamics model (perturbation-conditioned cell state evolution) + gene expression decoder. Captures temporal gene regulatory dynamics under perturbations.
> - **Intended Use:** In-silico virtual cell experiment engine. Predicts how gene expression changes over time in response to CRISPR knockouts, drug treatments, or combinatorial perturbations. Enables RL-based perturbation sequence planning.
> - **Training Data:** Replogle et al. genome-wide Perturb-seq (K562/RPE1), Norman et al. combinatorial CRISPRa, sci-Plex drug perturbations, developmental spatial transcriptomics.
> - **Architecture Details:** VAE encoder dimensions, Neural ODE solver type, perturbation conditioning mechanism, decoder architecture.
> - **Evaluation Metrics:** Mean squared error on held-out perturbation outcomes, correlation with ground-truth gene expression profiles, temporal trajectory accuracy.
> - **Limitations & Biases:** Limited to cell types present in training data. Combinatorial perturbation space is sparsely covered. Temporal resolution depends on available time-course data.
> - **How to Use:** Code snippet for loading from HF Hub, encoding a cell state, and predicting perturbation outcomes.
