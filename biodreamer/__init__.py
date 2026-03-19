"""
BioDreamer — World Models for Biological Design.

Top-level package that exposes the three domain modules and shared utilities:
    - biodreamer.core            — Shared base classes (world model, encoder, dynamics, etc.)
    - biodreamer.mol_dreamer     — MolDreamer: world models for molecular dynamics
    - biodreamer.protein_dreamer — ProteinDreamer: model-based RL for protein design
    - biodreamer.cell_dreamer    — CellDreamer: world models for gene regulatory dynamics
    - biodreamer.hub             — Hugging Face Hub integration (upload, download, registry)
    - biodreamer.training        — Training loops (world model, policy, active learning)
    - biodreamer.inference       — Inference pipelines (dreaming, candidate ranking)
    - biodreamer.utils           — Logging, metrics, visualisation, I/O helpers
"""
