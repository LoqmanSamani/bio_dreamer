/**
 * ProteinDreamerForm — Design Task Input Form.
 *
 * Purpose:
 *   Collects all parameters needed to submit a protein engineering
 *   dreaming job to the backend.
 *
 * Fields:
 *   - Wild-type sequence input (SequenceEditor component)
 *   - Target properties checkboxes: ΔΔG, Kd, kcat/Km, expression level
 *   - Property weights (sliders, 0-1 each)
 *   - Mutation budget: max number of mutations (1-20 slider)
 *   - Dreaming horizon: number of imagination steps (1-100)
 *   - Policy selector: dropdown (PPO, SAC, MCTS, ActiveInference)
 *   - World model selector: dropdown populated from /api/models
 *   - Structure file upload (optional PDB for structure-aware mode)
 *   - Advanced settings (collapsible):
 *       • Temperature, top-k, beam width
 *       • Uncertainty threshold for active learning
 *       • Random seed
 *
 * Validation:
 *   - Sequence must be valid amino acids, length 10-2000
 *   - At least one target property selected
 *   - Model must be loaded
 *
 * Actions:
 *   - "Dream" button → calls onSubmit(formData) → parent POSTs to API
 *   - "Reset" button → clears all fields to defaults
 */
