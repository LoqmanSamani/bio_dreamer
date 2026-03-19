# BioDreamer — System Architecture

> This document describes the high-level software architecture of the BioDreamer
> platform. It covers the three-tier web application (frontend → backend → ML core),
> the shared world-model abstraction, the Hugging Face Hub integration layer, and
> the asynchronous job system for long-running training / inference tasks.
>
> ## Sections to write
>
> 1. **High-Level Overview** — Three-tier architecture diagram (frontend, API server, ML workers).
> 2. **ML Core (`biodreamer/`)** — Shared base classes, domain-specific modules (MolDreamer, ProteinDreamer, CellDreamer), and how they plug into the unified world-model interface.
> 3. **Model Registry & Hugging Face Hub** — How models are versioned, stored locally, pushed to / pulled from Hugging Face Hub. Support for both community models (ESM-2, ProteinMPNN, etc.) and custom-trained BioDreamer models.
> 4. **Backend API (`server/`)** — FastAPI router structure, request/response schemas, authentication, job queue.
> 5. **Frontend (`frontend/`)** — Next.js app router, page structure, 3D viewers (Mol*/NGL), interactive forms.
> 6. **Async Job System** — How long-running tasks (world model training, imagination rollouts, active learning loops) are dispatched to GPU workers and tracked via the web UI.
> 7. **Data Flow Diagrams** — End-to-end flow for each module: user input → API → worker → model inference → result → frontend display.
> 8. **Deployment** — Docker Compose for local dev, Kubernetes for production, GPU resource management.
