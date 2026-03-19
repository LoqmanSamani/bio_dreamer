/**
 * MolViewer — 3-D Molecular Structure Viewer.
 *
 * Purpose:
 *   Interactive 3-D viewer for small molecules and molecular systems
 *   using the Mol* (Molstar) library. Used across MolDreamer and
 *   ProteinDreamer modules.
 *
 * Props:
 *   - structureData: string     — PDB/mmCIF/SDF content or URL
 *   - format: "pdb" | "mmcif" | "sdf" | "mol2"
 *   - representations?: Array   — visual styles (cartoon, ball-and-stick, surface)
 *   - highlights?: Array        — residues/atoms to highlight (e.g. mutation sites)
 *   - trajectory?: Array        — optional trajectory frames for playback
 *   - width?: number
 *   - height?: number
 *
 * Features:
 *   - Initialises Mol* plugin in a ref'd <div>
 *   - Supports loading from URL or inline data
 *   - Overlay controls: representation toggle, colour scheme, screenshot
 *   - Responsive resize via ResizeObserver
 *   - Trajectory playback with play/pause/scrubber when frames provided
 */
