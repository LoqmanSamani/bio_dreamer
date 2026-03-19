# Model Card: ProteinDreamer World Model

> Hugging Face-style model card for the ProteinDreamer world model — the core
> model of the BioDreamer project.
>
> ## Sections to write
>
> - **Model Description:** Joint PLM (ESM-2) + GVP-GNN encoder mapping (sequence, structure) → latent state. Conditional transformer/diffusion dynamics model predicting latent state transitions after mutations. Multi-task fitness decoder (ΔΔG, Kd, kcat). Trained on ProteinGym DMS assays + Tsuboyama mega-scale stability data.
> - **Intended Use:** Learned protein fitness landscape simulator for model-based RL. Enables "dreaming" — imagining multi-step mutation trajectories and their fitness outcomes without wet-lab experiments.
> - **Training Data:** ProteinGym (200+ DMS assays), Tsuboyama mega-scale (500k+ ΔΔG measurements), AlphaFold2/ESMFold predicted structures.
> - **Architecture Details:** ESM-2 650M backbone, GVP-GNN structure encoder, latent dimension, dynamics model type, reward head architecture.
> - **Evaluation Metrics:** Spearman ρ on held-out DMS assays, NDCG for top-variant ranking, rollout consistency over multi-step paths.
> - **Limitations & Biases:** Biased toward proteins with DMS data (mostly small, globular domains). Epistatic interactions beyond double mutants are underrepresented. Structure predictions are approximate.
> - **How to Use:** Code snippet for loading from HF Hub, encoding a wild-type, and running imagination rollouts.
