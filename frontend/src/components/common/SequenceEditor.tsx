/**
 * SequenceEditor — Interactive Protein Sequence Input & Editor.
 *
 * Purpose:
 *   A text editor for protein sequences (amino-acid one-letter codes).
 *   Supports manual editing, paste, file upload (FASTA), and highlights
 *   mutation positions relative to a reference sequence.
 *
 * Props:
 *   - sequence: string               — current amino acid sequence
 *   - onChange: (seq: string) => void — callback on edit
 *   - referenceSequence?: string      — wild-type for diff highlighting
 *   - readOnly?: boolean
 *   - maxLength?: number              — sequence length limit
 *
 * Features:
 *   - Monospace font, numbered residue positions
 *   - Character validation (only valid amino acids)
 *   - Colour-coded diff: mutations in red, insertions in blue, deletions in gray
 *   - FASTA file upload button (parses header + sequence)
 *   - Copy-to-clipboard button
 *   - Sequence statistics bar: length, molecular weight, pI estimate
 */
