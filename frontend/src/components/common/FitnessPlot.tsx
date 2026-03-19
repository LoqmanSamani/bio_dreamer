/**
 * FitnessPlot — Fitness Landscape Visualisation.
 *
 * Purpose:
 *   Interactive chart component for visualising fitness values over
 *   dreaming generations or mutation steps. Used primarily in
 *   ProteinDreamer but applicable to any module.
 *
 * Props:
 *   - data: Array<{generation: number, fitness: number, uncertainty?: number}>
 *   - paretoFront?: Array<{x: number, y: number}>  — Pareto-optimal points
 *   - objectives?: string[]        — axis labels for multi-objective plots
 *   - plotType?: "line" | "scatter" | "heatmap" | "pareto"
 *   - interactive?: boolean        — enable click-to-select points
 *   - onPointSelect?: (point) => void
 *
 * Features:
 *   - Line chart: fitness vs. generation with uncertainty bands (±σ)
 *   - Scatter: multi-objective trade-off with Pareto front overlay
 *   - Heatmap: 2-D fitness landscape (latent space projection)
 *   - Zoom, pan, tooltip with candidate details
 *   - Built on Recharts or Plotly.js (TBD)
 *   - Export as PNG / SVG
 */
