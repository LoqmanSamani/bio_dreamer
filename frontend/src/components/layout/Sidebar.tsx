/**
 * Sidebar — Left Navigation Panel.
 *
 * Purpose:
 *   Secondary navigation and context panel on the left side of the app.
 *   Content adapts based on the current module page.
 *
 * Sections:
 *   - Module-specific sub-navigation:
 *       • ProteinDreamer: "New Design", "Active Learning", "Landscapes"
 *       • MolDreamer: "New Simulation", "Trajectory Library"
 *       • CellDreamer: "New Plan", "Datasets", "GRN Explorer"
 *
 *   - Recent Jobs (last 5 for current module):
 *       • Status icon + job name + elapsed time
 *       • Click to navigate to results
 *
 *   - Quick Actions:
 *       • Upload data
 *       • Load model
 *       • View documentation
 *
 * Behaviour:
 *   - Collapsible (icon-only mode) with toggle button
 *   - Persists collapse state in localStorage
 *   - Hidden on mobile (content moves to hamburger menu)
 *   - Width: 260px expanded, 64px collapsed
 */
