"""
server.routers.cell_dreamer — CellDreamer API Endpoints.

Purpose:
    REST API endpoints for the CellDreamer module. Handles perturbation
    planning and cell state prediction jobs.

Endpoints to implement:
    POST /api/cell-dreamer/submit
        Request: initial expression profile (CSV/h5ad), target cell state,
                 perturbation budget, available perturbation library
        Response: job_id

    GET /api/cell-dreamer/results/{job_id}
        Response: planned perturbation sequence, predicted expression trajectory,
                  UMAP coordinates, gene heatmap data

    POST /api/cell-dreamer/predict
        Request: expression profile + specific perturbation
        Response: predicted post-perturbation expression (quick evaluation)

    GET /api/cell-dreamer/cell-types
        Response: list of available reference cell types for target specification

Design notes:
    - Expression data upload supports CSV (genes × cells) and h5ad (AnnData) formats.
    - Results include UMAP embedding coordinates showing the predicted cell state
      trajectory overlaid on a reference atlas.
    - Gene heatmap data is returned as a ranked list of differentially expressed genes.
"""
