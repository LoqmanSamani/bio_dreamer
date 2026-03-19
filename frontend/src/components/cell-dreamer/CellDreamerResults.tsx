/**
 * CellDreamerResults — Perturbation Results Display.
 *
 * Purpose:
 *   Renders output of a CellDreamer perturbation dreaming job.
 *   Shows ranked perturbation strategies, UMAP trajectories,
 *   gene expression changes, and GRN effects.
 *
 * Props:
 *   - jobId: string
 *   - results: PerturbationPlanResult  — from API schema
 *   - isLoading: boolean
 *
 * Sections:
 *   1. Ranked Perturbation Table:
 *      - Rank, perturbation (e.g. "CEBPA KO + 10nM dexamethasone"),
 *        predicted distance to target, confidence, expected effect size
 *      - Click to expand details
 *
 *   2. UMAP Trajectory Plot:
 *      - 2-D UMAP scatter: source cells, target cells, predicted trajectory
 *      - Colour-coded by cell type, arrows showing dreamed transitions
 *      - Interactive: hover shows cell metadata
 *
 *   3. Gene Expression Heatmap (GeneHeatmap component):
 *      - Top differentially expressed genes across perturbation steps
 *      - Clustered rows, time-point columns
 *
 *   4. Cell State Distance Plot:
 *      - Distance to target cell state over dreaming steps
 *      - Uncertainty bands
 *
 *   5. GRN View (optional):
 *      - Network graph of top affected gene regulators
 *      - Edge weights = regulatory strength changes
 */
