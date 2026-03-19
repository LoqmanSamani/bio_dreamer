/**
 * TrajectoryPlayer — MD Trajectory Playback Widget.
 *
 * Purpose:
 *   Animates a molecular dynamics or dreamed trajectory frame-by-frame
 *   in the 3-D viewer. Provides VCR-style playback controls.
 *
 * Props:
 *   - frames: Array<string>        — PDB snapshots per frame
 *   - fps?: number                 — playback speed (default 10)
 *   - currentFrame?: number        — controlled frame index
 *   - onFrameChange?: (idx: number) => void
 *   - properties?: Array<{frame: number, energy: number, rmsd: number}>
 *
 * Features:
 *   - Play / Pause / Step-forward / Step-backward buttons
 *   - Frame scrubber slider
 *   - Speed control (0.5× to 4×)
 *   - Synchronised line chart below showing energy/RMSD at current frame
 *   - Loop toggle
 *   - Export trajectory as multi-frame PDB
 */
