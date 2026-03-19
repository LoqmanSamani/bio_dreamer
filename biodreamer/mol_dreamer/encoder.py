"""
biodreamer.mol_dreamer.encoder — SE(3)-Equivariant Molecular Encoder.

Purpose:
    Encodes atomic-level molecular systems (3D coordinates, atom types, bond
    connectivity) into a compact latent representation z_t that respects
    rotational and translational symmetries of physical space.

Components to implement:
    - MolEncoder(BaseEncoder):
        Wraps an SE(3)-equivariant GNN (EGNN, PaiNN, or SchNet) that operates
        on molecular graphs where:
            - Nodes = atoms (features: element type, charge, hybridisation)
            - Edges = bonds + distance-based neighbours (features: distance, bond type)
        Produces a fixed-size latent vector z_t per molecular snapshot via
        global pooling over node embeddings.

    Architecture candidates:
        - EGNN (Satorras et al., 2021): simple, efficient E(n)-equivariant GNN
        - PaiNN (Schütt et al., 2021): message passing with equivariant vector features
        - SchNet (Schütt et al., 2017): continuous-filter convolutional layers

    Pre-trained options from Hugging Face / torch-geometric model zoo.

Design notes:
    - The encoder must produce latent states that are invariant to global rotation
      and translation (the physics doesn't change if you rotate the molecule).
    - Support both all-atom and coarse-grained (Martini 3) representations.
    - Batch processing of trajectory frames for efficient training.
"""
