/**
 * useJob — Job Lifecycle Hook.
 *
 * Purpose:
 *   Manages the full lifecycle of a background job: submission, polling
 *   for status, SSE log streaming, and result retrieval.
 *
 * Parameters:
 *   - jobId?: string            — existing job to track (optional)
 *   - pollInterval?: number     — ms between status polls (default 2000)
 *
 * Returns:
 *   {
 *     jobId,              — assigned job ID after submission
 *     status,             — "queued" | "running" | "completed" | "failed" | "cancelled"
 *     progress,           — 0-100 percentage
 *     logs,               — string[] of log lines (from SSE)
 *     result,             — parsed result payload (when completed)
 *     error,              — error message (when failed)
 *     submit(endpoint, data), — POST to endpoint, start tracking
 *     cancel(),           — POST /api/jobs/{id}/cancel
 *   }
 *
 * Behaviour:
 *   - On submit: POSTs data, stores jobId, starts polling
 *   - Opens SSE connection to /api/jobs/{id}/logs for live log streaming
 *   - Stops polling when terminal state reached (completed/failed/cancelled)
 *   - Cleans up SSE and polling on component unmount
 */
