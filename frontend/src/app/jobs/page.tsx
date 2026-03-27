export default function JobsPage() {
  return <div />;
}

/**
 * Jobs Page — Task Queue Dashboard.
 *
 * Purpose:
 *   Monitor and manage all submitted jobs (dreaming runs, training tasks,
 *   active-learning cycles). Provides real-time status updates via SSE
 *   (Server-Sent Events).
 *
 * Features:
 *   - Filterable/sortable job table:
 *       • Columns: ID, module, type, status, submitted, elapsed, progress
 *       • Status badges: queued, running, completed, failed, cancelled
 *       • Filter by module (MolDreamer / ProteinDreamer / CellDreamer)
 *   - Job detail drawer (click to expand):
 *       • Live log stream (SSE from /api/jobs/{id}/logs)
 *       • Progress bar with step count
 *       • "Cancel" button for running jobs
 *       • "View Results" link to module results page
 *   - Pagination and auto-refresh toggle
 *
 * Data fetching:
 *   - GET    /api/jobs              → list jobs (paginated)
 *   - GET    /api/jobs/{id}         → job detail
 *   - POST   /api/jobs/{id}/cancel  → cancel running job
 *   - GET    /api/jobs/{id}/logs    → SSE log stream
 */
