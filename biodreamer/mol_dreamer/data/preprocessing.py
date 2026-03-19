"""
biodreamer.mol_dreamer.data.preprocessing — MD Trajectory Preprocessing.

Purpose:
    Preprocessing utilities for converting raw MD trajectory files into
    graph-structured tensors suitable for the SE(3)-equivariant encoder.

Components to implement:
    - pdb_to_graph(pdb_path) → PyG Data object (nodes=atoms, edges=bonds+neighbours)
    - trajectory_to_frames(traj_path, topology_path, stride) → list of coordinate tensors
    - compute_features(coords, atom_types) → node features, edge features
    - align_frames(frames, reference) → RMSD-aligned coordinate frames
    - coarse_grain(all_atom_coords, mapping) → CG bead coordinates (Martini-style)
    - normalise_coordinates(coords) → centred, optionally scaled coordinates

Design notes:
    - Use MDAnalysis or MDTraj for trajectory I/O.
    - Graph construction uses torch_geometric Data objects.
    - Neighbour edges based on distance cutoff (e.g., 5Å).
    - Preprocessing should be cached to disk for fast re-loading.
"""
