# BioDreamer — Training Guide

> How to train world models and RL policies for each BioDreamer module.
>
> ## Sections to write
>
> 1. **Data Preparation** — How to download and preprocess training data for each module.
>    - ProteinDreamer: ProteinGym DMS assays, Tsuboyama mega-scale data, ESMFold structures.
>    - MolDreamer: MD trajectory datasets (ATLAS, DE Shaw), PDB structures.
>    - CellDreamer: Perturb-seq datasets (Replogle, Norman), scRNA-seq preprocessing.
>
> 2. **World Model Training** — Training the encoder + dynamics + decoder + reward head.
>    - Configuration via YAML files in configs/.
>    - Distributed training with PyTorch DDP.
>    - Logging to Weights & Biases.
>    - Checkpointing and model selection.
>
> 3. **Policy Training** — Training the RL agent (SAC/PPO/MCTS) inside the learned world model.
>    - Imagination rollouts (dreaming).
>    - Active Inference vs. standard RL objective comparison.
>    - Curriculum learning (single mutations → multi-step trajectories).
>
> 4. **Active Learning Loop** — How to integrate real experimental feedback.
>    - World model fine-tuning on new data.
>    - Uncertainty-driven candidate selection.
>
> 5. **Exporting to Hugging Face Hub** — Push trained models with auto-generated model cards.
>
> 6. **Hyperparameter Tuning** — Key hyperparameters for each module and recommended search ranges.
