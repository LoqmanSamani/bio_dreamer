"""
biodreamer.protein_dreamer.decoder — Protein State Decoder.

Purpose:
    Reconstructs interpretable outputs from the protein latent state z_t:
    mutated sequence, predicted structure changes, per-residue confidence.

Components to implement:
    - ProteinDecoder(BaseDecoder):
        - decode_sequence(z_t) → amino acid probabilities per position (L × 20)
        - decode_structure(z_t) → predicted Cα coordinates or distance map
        - decode_plddt(z_t) → per-residue confidence (pLDDT-like)
        - decode_all(z_t) → dict with sequence, structure, confidence

Design notes:
    - The sequence decoder essentially performs "inverse folding in latent space" —
      it predicts what sequence corresponds to the current latent representation.
    - Structure decoding is approximate — for high-fidelity structure, the decoded
      sequence is passed through ESMFold/AF2 as a validation oracle.
    - Decoder outputs are displayed in the web frontend: sequence alignment view,
      3D structure viewer (Mol*), per-residue pLDDT heatmap.
    - Training loss: sequence cross-entropy + structure coordinate MSE + pLDDT MSE.
"""

import numpy as np
import torch

