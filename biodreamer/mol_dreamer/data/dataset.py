"""
biodreamer.mol_dreamer.data.dataset — MD Trajectory Datasets.

Purpose:
    PyTorch Dataset classes for loading molecular dynamics trajectory data
    and presenting it as (state, action, next_state, reward) tuples for
    world model training.

Components to implement:
    - MDTrajectoryDataset(torch.utils.data.Dataset):
        Loads MD trajectory files (XTC, DCD, PDB frames) and extracts
        consecutive frame pairs as training transitions.
        Returns: (coords_t, coords_{t+1}, Δt, properties_t)

    - ATLASDataset(MDTrajectoryDataset):
        Loader for the ATLAS protein dynamics dataset (many short MD
        trajectories of diverse proteins).

    - DEShawDataset(MDTrajectoryDataset):
        Loader for DE Shaw long-timescale MD trajectories (BPTI, ubiquitin).

    - PDBBindDataset:
        Loads protein-ligand complexes with experimental binding affinities.
        Used for reward head training.

Design notes:
    - Trajectories should be loaded lazily (memory-mapped or chunked) for
      large-scale training.
    - Support for both all-atom and coarse-grained representations.
    - Frame-pair sampling strategies: consecutive, strided (for learning
      multi-timescale dynamics), random.
"""
