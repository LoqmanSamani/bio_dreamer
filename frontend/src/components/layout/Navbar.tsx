/**
 * Navbar — Top Navigation Bar.
 *
 * Purpose:
 *   Persistent horizontal navigation bar at the top of the application.
 *
 * Elements:
 *   - Left: BioDreamer logo + wordmark (links to /)
 *   - Centre: Module navigation tabs:
 *       • MolDreamer   (emerald accent)
 *       • ProteinDreamer (indigo accent)
 *       • CellDreamer  (amber accent)
 *       • Model Hub
 *       • Jobs
 *       • Blog (teal accent)
 *     Active tab is highlighted with module-specific colour
 *   - Right:
 *       • Backend status indicator (green dot = healthy)
 *       • GPU memory usage badge
 *       • Dark/light mode toggle (persisted in localStorage)
 *
 * Behaviour:
 *   - Uses Next.js `usePathname()` to determine active tab
 *   - Responsive: collapses to hamburger menu on mobile
 *   - Sticky positioning (top: 0, z-50)
 */
