"""
biodreamer.mol_dreamer.decoder — Molecular State Decoder.

Purpose:
    Reconstructs physical observables from latent molecular states:
        z_t → (3D coordinates, contact maps, RMSD, RMSF, secondary structure)

Components to implement:
    - MolDecoder(BaseDecoder):
        - decode_coordinates(z_t) → atomic Cartesian coordinates (N_atoms × 3)
        - decode_contacts(z_t) → predicted residue–residue or atom–atom contact map
        - decode_observables(z_t) → dict of scalar observables (RMSD, Rg, SASA, etc.)

Design notes:
    - Coordinate reconstruction should respect physical constraints (bond lengths,
      angles) — consider using an equivariant decoder or post-hoc energy minimisation.
    - The decoder is used for visualisation in the web frontend (trajectory player)
      and for validation against ground-truth MD snapshots during training.
    - Loss: coordinate RMSD + contact map BCE + observable MSE.
"""
