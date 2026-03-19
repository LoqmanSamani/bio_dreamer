---
title: "Introducing BioDreamer: World Models for Biological Design"
date: 2026-03-19
author: "Loqman Samani"
tags: [announcement, world-models, protein-dreamer, mol-dreamer, cell-dreamer]
summary: "Why we're building BioDreamer — a unified framework that teaches machines to dream about biology so we don't have to wait for every experiment."
---

# Introducing BioDreamer: World Models for Biological Design

Biology operates at every scale — from atoms jiggling in a protein's binding pocket, to amino-acid mutations reshaping fitness landscapes, to thousands of genes orchestrating cell fate. At each scale, the core challenge is the same: **the real environment is prohibitively expensive to query**.

- A molecular dynamics simulation of one protein: **days on a GPU cluster**
- One round of directed evolution: **months and thousands of dollars**
- A genome-wide CRISPR screen: **millions of cells, weeks of work**

Yet intelligent design demands exploring vast combinatorial spaces. Current ML approaches are mostly **one-shot** — generate a candidate, hope it works, repeat. No existing system does what a skilled engineer would: **plan multi-step strategies by mentally simulating outcomes before committing**.

## The World Model Approach

BioDreamer applies *world models* from model-based reinforcement learning to biological systems. The core idea:

1. **Learn** a latent-space simulator of the biological environment from data
2. **Dream** — roll out imagined trajectories inside the learned model
3. **Plan** optimal interventions (mutations, perturbations) in imagination
4. **Act** — only run the most promising experiments in the real world
5. **Update** the model with new results and repeat

This replaces brute-force experimentation with intelligent, amortised, in-silico reasoning.

## Three Modules

**MolDreamer** learns molecular dynamics in latent space. Instead of integrating Newton's equations step-by-step, it jumps directly between interesting conformational states, predicting binding free energy and stability along the way.

**ProteinDreamer** navigates protein fitness landscapes via dreaming. Given a wild-type protein and target properties (stability, affinity, catalytic activity), it plans multi-step mutation strategies by simulating their effects in the world model — guided by Active Inference for principled exploration.

**CellDreamer** plans cellular reprogramming strategies. Given a source cell state and a target (e.g., convert fibroblast → cardiomyocyte), it dreams optimal perturbation sequences (gene knockouts, drug treatments) using a world model of gene regulatory dynamics.

## What's Next

- Building the core world model architecture (DreamerV3-style RSSM)
- Training ProteinDreamer on ProteinGym benchmarks
- Integrating ESM-2 and ESMFold as encoder backbones
- Setting up the active learning loop for wet-lab feedback
- Launching the web interface for interactive protein design

Follow the project on [GitHub](https://github.com/LoqmanSamani/bio_dreamer) for updates.
