/**
 * useModel — Model Management Hook.
 *
 * Purpose:
 *   Manages model listing, loading, and unloading from the frontend.
 *   Provides reactive state for the currently loaded models and
 *   available models from both HF Hub and local checkpoints.
 *
 * Returns:
 *   {
 *     models,             — Array of ModelInfo (all available)
 *     loadedModels,       — Array of ModelInfo (currently on GPU)
 *     isLoading,          — boolean
 *     loadModel(id),      — POST /api/models/load
 *     unloadModel(id),    — POST /api/models/unload
 *     downloadModel(hfId),— POST /api/models/download
 *     uploadModel(id),    — POST /api/models/upload
 *     refreshModels(),    — refetch model list
 *     gpuMemoryUsage,     — { used: number, total: number } in MB
 *   }
 *
 * Behaviour:
 *   - Fetches model list on mount and caches
 *   - Optimistic UI updates on load/unload
 *   - Polls GPU memory usage every 10s while models are loaded
 */
