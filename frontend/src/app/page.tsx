/**
 * Dashboard / Landing Page — BioDreamer Home.
 *
 * Purpose:
 *   The main entry point users see when opening the application.
 *   Provides an overview of all three BioDreamer modules and quick
 *   access to each one.
 *
 * Components to render:
 *   - Hero section with BioDreamer branding and tagline
 *   - Three module cards (MolDreamer, ProteinDreamer, CellDreamer):
 *       • Module name, icon/illustration, brief description
 *       • Status indicator (loaded model count, GPU availability)
 *       • "Launch" button navigating to the module page
 *   - Recent Jobs panel — last 5 jobs across all modules with status badges
 *   - System status footer — backend health, loaded models, GPU memory usage
 *
 * Data fetching:
 *   - GET /api/health         → backend status
 *   - GET /api/models         → loaded model summary
 *   - GET /api/jobs?limit=5   → recent jobs
 */
