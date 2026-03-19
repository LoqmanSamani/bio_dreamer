/**
 * FileUploader — Drag-and-Drop File Upload Component.
 *
 * Purpose:
 *   Generic file upload widget supporting drag-and-drop and click-to-browse.
 *   Used across all modules for uploading PDB files, FASTA sequences,
 *   h5ad scRNA-seq data, and other biological data files.
 *
 * Props:
 *   - accept: string[]              — allowed MIME types or extensions
 *                                      e.g. [".pdb", ".fasta", ".h5ad", ".csv"]
 *   - maxSizeMB?: number            — max file size (default 100 MB)
 *   - multiple?: boolean            — allow multiple files
 *   - onUpload: (files: File[]) => void
 *   - onError?: (error: string) => void
 *   - label?: string                — instructional text
 *   - icon?: ReactNode              — custom upload icon
 *
 * Features:
 *   - Drag-and-drop zone with visual feedback (border highlight)
 *   - File type validation with user-friendly error messages
 *   - File size validation
 *   - Preview: file name, size, type badge after selection
 *   - Remove button per file
 *   - Upload progress bar (when paired with API call)
 */
