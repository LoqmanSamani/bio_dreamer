# BioDreamer — API Reference

> Auto-generated and manually curated reference for the BioDreamer REST API.
> This document covers all endpoints exposed by the FastAPI backend.
>
> ## Endpoint groups to document
>
> 1. **Protein Dreamer API** (`/api/protein-dreamer/`)
>    - POST /submit     — Submit a protein design job (wild-type sequence + objective)
>    - GET  /results    — Retrieve dream trajectories, ranked candidates, fitness plots
>    - POST /evaluate   — Run single-candidate evaluation against structure oracle
>
> 2. **Mol Dreamer API** (`/api/mol-dreamer/`)
>    - POST /submit     — Submit a molecular dynamics world-model job
>    - GET  /results    — Retrieve latent trajectories, property predictions
>
> 3. **Cell Dreamer API** (`/api/cell-dreamer/`)
>    - POST /submit     — Submit a perturbation planning job
>    - GET  /results    — Retrieve predicted cell state trajectories
>
> 4. **Models API** (`/api/models/`)
>    - GET  /list       — List available models (local + Hugging Face Hub)
>    - POST /load       — Load a model into memory
>    - POST /upload     — Push a trained model to Hugging Face Hub
>
> 5. **Jobs API** (`/api/jobs/`)
>    - GET  /           — List all jobs for the current session
>    - GET  /{job_id}   — Get status and partial results of a specific job
>    - POST /{job_id}/cancel — Cancel a running job
