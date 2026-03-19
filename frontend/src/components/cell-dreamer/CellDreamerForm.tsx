/**
 * CellDreamerForm — Perturbation Planning Input Form.
 *
 * Purpose:
 *   Collects parameters for a CellDreamer perturbation dreaming job.
 *
 * Fields:
 *   - Dataset input:
 *       • Upload scRNA-seq h5ad file (FileUploader)
 *       • OR select preloaded dataset (Norman 2019, SciPlex, custom)
 *   - Source cell state:
 *       • Dropdown populated from dataset cell types
 *       • OR upload reference expression profile
 *   - Target cell state:
 *       • Dropdown of available cell types
 *       • OR define by marker gene expression targets
 *   - Perturbation type:
 *       • Radio: gene knockout, drug treatment, overexpression, combinatorial
 *   - Budget:
 *       • Max perturbations: slider (1-10)
 *       • Dreaming horizon: slider (1-50 steps)
 *   - Model selector: dropdown from /api/models
 *   - Advanced (collapsible):
 *       • Dosage levels (for drug mode)
 *       • Gene candidates whitelist/blacklist
 *       • Random seed
 *
 * Actions:
 *   - "Plan Perturbation" → onSubmit(formData)
 *   - "Reset" → defaults
 */
