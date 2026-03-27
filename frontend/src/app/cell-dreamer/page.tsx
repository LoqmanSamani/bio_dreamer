export default function CellDreamerPage() {
  return <div />;
}

/**
 * CellDreamer Module Page.
 *
 * Purpose:
 *   Interface for cellular reprogramming via perturbation world models.
 *   Users define a source cell state, a target cell state, and the model
 *   dreams optimal perturbation strategies (gene knockouts, drug treatments,
 *   combinations) to steer the transcriptome.
 *
 * Layout:
 *   Left panel — Input & Controls:
 *     - CellDreamerForm: upload scRNA-seq data (h5ad) or select from
 *       preloaded datasets (Norman 2019, SciPlex), define source/target
 *       cell types, select perturbation type (knockout, drug, combo),
 *       set budget and dreaming horizon
 *     - Model selector: pick world-model checkpoint or HF Hub model
 *     - "Plan Perturbation" button → POST /api/cell-dreamer/submit
 *
 *   Right panel — Results & Visualisation:
 *     - CellDreamerResults: ranked perturbation plans, UMAP trajectory
 *       (source → predicted → target), gene expression heatmap (top DEGs),
 *       cell state distance plot over dream steps
 *     - Gene Regulatory Network (GRN) view for top perturbations
 *
 * Data fetching:
 *   - POST /api/cell-dreamer/submit
 *   - GET  /api/cell-dreamer/results/{job_id}
 *   - POST /api/cell-dreamer/predict
 *   - GET  /api/cell-dreamer/cell-types
 */
