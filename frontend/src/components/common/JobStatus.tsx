/**
 * JobStatus — Job Status Badge & Mini-Card Component.
 *
 * Purpose:
 *   Reusable component that displays the current status of a background
 *   job. Used in the Jobs page table, Sidebar recent-jobs list, and
 *   inline within module result pages.
 *
 * Props:
 *   - jobId: string
 *   - status: "queued" | "running" | "completed" | "failed" | "cancelled"
 *   - progress?: number            — 0-100 percentage
 *   - module?: "protein-dreamer" | "mol-dreamer" | "cell-dreamer"
 *   - elapsedTime?: number         — seconds since job started
 *   - compact?: boolean            — badge-only mode vs. expanded card
 *   - onCancel?: () => void        — optional cancel handler
 *
 * Visual:
 *   - Colour-coded badge:
 *       queued=gray, running=blue (animated pulse), completed=green,
 *       failed=red, cancelled=yellow
 *   - Progress bar (when running)
 *   - Elapsed time in human-readable format (e.g. "2m 34s")
 *   - Module icon for context
 */
