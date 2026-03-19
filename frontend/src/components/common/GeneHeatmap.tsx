/**
 * GeneHeatmap — Gene Expression Heatmap for CellDreamer.
 *
 * Purpose:
 *   Displays a heatmap of gene expression values across conditions,
 *   time points, or perturbations. Primary visualisation for CellDreamer
 *   results showing differentially expressed genes.
 *
 * Props:
 *   - matrix: number[][]            — expression matrix (genes × conditions)
 *   - geneNames: string[]           — row labels
 *   - conditionNames: string[]      — column labels
 *   - colorScale?: "viridis" | "RdBu" | "plasma"  — colour map
 *   - clusterRows?: boolean         — hierarchical clustering of genes
 *   - clusterCols?: boolean         — hierarchical clustering of conditions
 *   - selectedGenes?: string[]      — genes to highlight
 *   - onGeneSelect?: (gene: string) => void
 *
 * Features:
 *   - Canvas-based rendering for performance (up to 2000 genes)
 *   - Dendrogram for clustered axes
 *   - Tooltip: gene name, condition, expression value, fold change
 *   - Row/column selection and zoom
 *   - Colour legend with min/max values
 *   - Export as SVG or CSV
 */
