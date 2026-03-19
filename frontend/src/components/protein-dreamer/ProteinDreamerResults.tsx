/**
 * ProteinDreamerResults — Design Results Display.
 *
 * Purpose:
 *   Renders the output of a ProteinDreamer dreaming job. Shows ranked
 *   candidate variants, fitness plots, structure comparisons, and
 *   active-learning suggestions.
 *
 * Props:
 *   - jobId: string
 *   - results: ProteinDesignResult  — from API schema
 *   - isLoading: boolean
 *
 * Sections:
 *   1. Candidate Table:
 *      - Rank, mutations (e.g. "A42G/L55V"), predicted fitness, uncertainty,
 *        Pareto-optimal flag
 *      - Sortable by any column
 *      - Click row → expand to show structure + details
 *
 *   2. Fitness Evolution Chart (FitnessPlot component):
 *      - Best/mean/worst fitness per dreaming generation
 *      - Uncertainty bands
 *
 *   3. Structure Viewer (ProteinViewer component):
 *      - Selected candidate's predicted structure
 *      - Mutation sites highlighted
 *      - Side-by-side with wild-type (split view)
 *
 *   4. Mutation Trajectory Timeline:
 *      - Horizontal timeline showing order of mutations introduced
 *      - Fitness gain per step
 *
 *   5. Active Learning Panel:
 *      - "Propose Experiments" button
 *      - Ranked list of suggested wet-lab validations
 *      - Expected information gain per suggestion
 */
