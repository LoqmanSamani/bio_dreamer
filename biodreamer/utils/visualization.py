"""
biodreamer.utils.visualization — Plotting and Visualisation Utilities.

Purpose:
    Generates plots and visual data for training monitoring, evaluation
    reports, and the web frontend. Uses matplotlib for static plots and
    returns JSON-serialisable data for interactive frontend visualisations.

Components to implement:
    - plot_fitness_landscape(predictions, ground_truth) → matplotlib Figure
    - plot_training_curves(history) → loss and metric curves
    - plot_mutation_trajectory(trajectory) → sequence alignment + fitness path
    - plot_latent_space(embeddings, labels) → UMAP/t-SNE visualisation
    - plot_uncertainty_calibration(predicted_std, errors) → calibration plot
    - plot_gene_heatmap(expression_matrix, gene_names) → heatmap Figure
    - trajectory_to_json(trajectory) → dict for frontend TrajectoryPlayer
    - structure_to_json(coordinates, plddt) → dict for frontend ProteinViewer
    - expression_to_json(expression_profile, gene_names) → dict for frontend GeneHeatmap

Design notes:
    - Static plots (matplotlib) are used in notebooks and training scripts.
    - JSON-serialisable outputs are used by the FastAPI server → frontend pipeline.
    - The frontend uses Mol* for 3D protein visualisation — this module provides
      the data in Mol*-compatible format (PDB string or mmCIF).
"""
