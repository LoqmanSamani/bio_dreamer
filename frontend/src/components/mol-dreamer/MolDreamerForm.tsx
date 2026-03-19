/**
 * MolDreamerForm — Molecular Simulation Input Form.
 *
 * Purpose:
 *   Collects parameters for submitting a MolDreamer world-model
 *   simulation job.
 *
 * Fields:
 *   - Molecular system input:
 *       • Upload PDB/mol2/SDF file (FileUploader)
 *       • OR paste PDB ID for auto-download
 *   - Simulation parameters:
 *       • Temperature (K): slider 200-500
 *       • Pressure (atm): input field
 *       • Simulation length (ns): slider 1-1000
 *       • Time step (fs): dropdown [0.5, 1.0, 2.0]
 *   - Dreaming parameters:
 *       • Dreaming horizon (steps): slider
 *       • Coarse-graining level: dropdown [all-atom, Cα-only, residue-level]
 *   - Property targets (optional):
 *       • Binding free energy (ΔG), RMSD constraint, SASA range
 *   - World model selector: dropdown from /api/models
 *
 * Actions:
 *   - "Simulate" button → onSubmit(formData)
 *   - "Reset" → restore defaults
 */
