"""
biodreamer.cell_dreamer.data.dataset — Perturbation scRNA-seq Datasets.

Purpose:
    PyTorch Dataset classes for loading single-cell perturbation data
    (Perturb-seq, CRISPRi/a screens, drug perturbation experiments)
    as training data for the CellDreamer world model.

Components to implement:
    - PerturbSeqDataset(torch.utils.data.Dataset):
        Loads Perturb-seq experiments where cells receive CRISPR perturbations
        and expression is measured via scRNA-seq.
        Each sample: (control_expression, perturbation_id, perturbed_expression)
        Sources: Replogle et al. (2022) genome-wide CRISPRi in K562/RPE1

    - NormanDataset(PerturbSeqDataset):
        Specifically loads Norman et al. (2019) combinatorial CRISPRa data.
        Key: contains double-perturbation data for epistasis evaluation.

    - SciPlexDataset(torch.utils.data.Dataset):
        Loads sci-Plex drug perturbation data (compound + dose → expression).
        Each sample: (control_expression, drug_id, dose, perturbed_expression)

    - TimeCourseDataset:
        For time-resolved perturbation experiments (rare but valuable).
        Returns full trajectories: (expression_t0, perturbation, expression_t1, ..., expression_tN)

Design notes:
    - scRNA-seq data is stored as AnnData (.h5ad) files — use scanpy for loading.
    - Expression values: log-normalised counts per cell, after QC filtering.
    - Highly variable gene selection should be configurable (2k or 5k HVGs).
    - Control cells (no perturbation) are critical for baseline comparison.
"""
