#!/usr/bin/env bash
# download_data.sh — Dataset Download Script.
#
# Purpose:
#   Downloads all required datasets for training and evaluation of
#   BioDreamer modules. Creates the data/ directory structure and
#   fetches files from public repositories.
#
# Datasets to download:
#
#   ProteinDreamer:
#     - ProteinGym substitution benchmarks (CSV from GitHub)
#     - Tsuboyama et al. 2023 mega-scale DMS dataset
#     - UniProt reference sequences for target proteins
#
#   MolDreamer:
#     - ATLAS MD trajectory database (selected systems)
#     - D.E. Shaw long-timescale trajectories (BPTI, villin, etc.)
#     - PDBBind refined set (for binding affinity)
#
#   CellDreamer:
#     - Norman et al. 2019 Perturb-seq (h5ad from Figshare)
#     - SciPlex drug perturbation dataset
#     - Tabula Sapiens reference atlas (subset)
#
# Usage:
#   bash scripts/download_data.sh [--protein] [--mol] [--cell] [--all]
#
# Output:
#   data/
#     protein_dreamer/   — ProteinGym CSVs, structures
#     mol_dreamer/       — trajectory files, PDB structures
#     cell_dreamer/      — h5ad files, gene lists
