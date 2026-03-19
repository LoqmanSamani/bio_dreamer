/**
 * MolDreamerResults — Simulation Results Display.
 *
 * Purpose:
 *   Renders output of a MolDreamer dreaming/simulation job including
 *   3-D trajectory visualisation, energy landscapes, and property
 *   evolution plots.
 *
 * Props:
 *   - jobId: string
 *   - results: MolSimulationResult  — from API schema
 *   - isLoading: boolean
 *
 * Sections:
 *   1. 3-D Trajectory Viewer (MolViewer + TrajectoryPlayer):
 *      - Animated playback of dreamed conformational trajectory
 *      - Play/pause/scrub controls
 *
 *   2. Energy Plot:
 *      - Potential energy over time/frames
 *      - Kinetic energy overlay (optional)
 *      - Key frame annotations (minima, transitions)
 *
 *   3. Structural Metrics:
 *      - RMSD vs. time plot
 *      - RMSF per residue bar chart
 *      - Radius of gyration over time
 *
 *   4. Free Energy Surface:
 *      - 2-D contour plot projected onto top-2 PCA components
 *      - Metastable state labels
 *      - Transition pathways
 *
 *   5. Comparison Panel (if ground-truth available):
 *      - Dreamed vs. actual MD overlay
 *      - Correlation plots per property
 */
