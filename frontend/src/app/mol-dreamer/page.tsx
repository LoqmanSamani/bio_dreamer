export default function MolDreamerPage() {
  return <div />;
}

/**
 * MolDreamer Module Page.
 *
 * Purpose:
 *   Interface for molecular dynamics world-model exploration. Users upload
 *   or select a molecular system, configure simulation parameters, and
 *   launch dream rollouts that predict conformational dynamics without
 *   running full MD simulations.
 *
 * Layout:
 *   Left panel — Input & Controls:
 *     - MolDreamerForm: upload PDB/mol2 file or paste PDB ID, select
 *       force-field, temperature, simulation length, dreaming horizon
 *     - Property targets: binding free energy (ΔG), melting temperature (Tm),
 *       SASA, RMSD constraints
 *     - Model selector: choose world-model checkpoint or HF Hub model
 *     - "Simulate" button → POST /api/mol-dreamer/submit
 *
 *   Right panel — Results & Visualisation:
 *     - MolDreamerResults: 3-D molecular viewer (Mol*) with trajectory
 *       playback, energy landscape plot, RMSD/RMSF over time, free energy
 *       surface (2-D projection)
 *     - Dreamed vs. ground-truth comparison (when oracle data available)
 *
 * Data fetching:
 *   - POST /api/mol-dreamer/submit
 *   - GET  /api/mol-dreamer/results/{job_id}
 *   - POST /api/mol-dreamer/simulate
 */
