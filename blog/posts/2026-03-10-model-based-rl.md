---
title: "World Models: From Tabular Planning to Real-Time World Generation"
date: 2026-03-10
author: "Loghman Samani"
tags: [model-based-rl, world-models, dreamer, theory]
summary: "A survey of three decades of model-based reinforcement learning — from Sutton's Dyna to DeepMind's Genie 3 — covering 26 papers that trace the evolution of agents that learn to plan by dreaming."
externalUrl: "https://loqmansamani.github.io/articles/model_based_rl/index.html"
---
# World Models

*From tabular planning to real-time interactive world generation, three decades of model-based reinforcement learning.*

The core problem of reinforcement learning is deceptively simple. An agent interacts with an environment, receives rewards, and learns a policy that maximizes long-term return. Model-free methods attack this through trial and error — they work, eventually, but they are profligate with data. Model-based RL takes a fundamentally different approach: the agent first learns a model of the environment and then uses that model to plan, generate synthetic data, or compute gradients for policy improvement.

This article surveys that trajectory through **26 papers**, following the thread from Sutton's Dyna to DeepMind's Genie 3.

## What's Covered

- **Planning from Experience** — Dyna, prioritized sweeping, and the foundations of model-based planning
- **Taming Model Bias** — PILCO, SVG, and probabilistic dynamics models that express honest uncertainty
- **Deep Dynamics** — Neural network dynamics, Ha & Schmidhuber's World Models, MBPO, and PETS
- **Learning to Dream** — PlaNet's RSSM, the Dreamer trilogy (V1 → V2 → V3), and learning behaviors inside latent world models
- **Transformers, Diffusion, and Foundation Models** — IRIS, TD-MPC, DIAMOND, UniSim, and the Genie series
- **From Imagination to Reality** — DayDreamer on physical robots, UniPi's policy-as-video
- **The Neural Engine Era** — GameNGen, Oasis, NVIDIA Cosmos, V-JEPA 2

## Why This Matters for BioDreamer

BioDreamer is built directly on the principles surveyed here. By learning a world model of biological environments, we can plan optimal interventions — mutations, perturbations, reprogramming strategies — in imagination, dramatically reducing the need for expensive real-world experiments. The Dreamer architecture (RSSM + actor-critic in latent space) forms the computational backbone of all three BioDreamer modules.

---

📖 **[Read the full article →](https://loqmansamani.github.io/articles/model_based_rl/index.html)**
