export default function ModelsPage() {
  return <div />;
}

/**
 * Model Hub Page.
 *
 * Purpose:
 *   Browse, load, and manage models from both Hugging Face Hub and
 *   locally trained checkpoints. Serves as the central model management
 *   interface for all three BioDreamer modules.
 *
 * Sections:
 *   1. Community Models (from HF Hub):
 *      - Searchable/filterable table of pre-registered models
 *        (ESM-2, ESMFold, Geneformer, SchNet, etc.)
 *      - "Download" button, version selector, model card preview
 *
 *   2. Custom Models (locally trained):
 *      - List of user-trained world-model and policy checkpoints
 *      - Training metrics summary, upload-to-HF button
 *
 *   3. Loaded Models (currently in GPU memory):
 *      - Active model list with GPU memory usage per model
 *      - "Load" / "Unload" controls
 *      - Device assignment (GPU index selection)
 *
 * Data fetching:
 *   - GET    /api/models                → list all models
 *   - POST   /api/models/load           → load model to GPU
 *   - POST   /api/models/unload         → unload model
 *   - POST   /api/models/upload         → push to HF Hub
 *   - POST   /api/models/download       → pull from HF Hub
 */
