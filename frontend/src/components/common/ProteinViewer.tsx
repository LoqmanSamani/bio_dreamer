/**
 * ProteinViewer — Protein Structure Viewer with Mutation Annotations.
 *
 * Purpose:
 *   Specialised wrapper around MolViewer for protein structures.
 *   Adds protein-specific overlays: residue colouring by fitness,
 *   mutation site highlighting, pLDDT confidence colouring.
 *
 * Props:
 *   - pdbData: string           — PDB content of the protein structure
 *   - mutations?: Array<{position: number, wildtype: string, mutant: string}>
 *   - fitnessPerResidue?: number[]  — per-residue fitness values for heatmap
 *   - plddt?: number[]          — pLDDT confidence per residue (AlphaFold-style)
 *   - colorScheme?: "fitness" | "plddt" | "secondary" | "chain" | "bfactor"
 *   - showSurface?: boolean
 *
 * Features:
 *   - Highlights mutation sites with red spheres or coloured residues
 *   - Tooltip on hover: residue name, position, fitness, ΔΔG
 *   - Toggle between cartoon, surface, and ball-and-stick
 *   - Side-by-side comparison mode (wild-type vs. variant) via split view
 */
