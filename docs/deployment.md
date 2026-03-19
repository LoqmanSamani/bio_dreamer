# BioDreamer — Deployment Guide

> Instructions for deploying the BioDreamer platform in different environments:
> local development, single-server, and production (Kubernetes).
>
> ## Sections to write
>
> 1. **Prerequisites** — Python 3.10+, Node.js 18+, Docker, NVIDIA GPU drivers + CUDA.
> 2. **Local Development** — `make install && make dev` workflow.
> 3. **Docker Compose** — Single-machine deployment with docker-compose.yml.
> 4. **Production (K8s)** — Helm chart, GPU node pools, persistent volume claims for model cache.
> 5. **Environment Variables** — Reference to .env.example with explanations.
> 6. **Hugging Face Hub Setup** — How to configure HF_TOKEN, create an organisation, push/pull models.
> 7. **SSL / HTTPS** — Reverse proxy (nginx/Caddy) configuration for production.
> 8. **Monitoring** — Prometheus + Grafana for server health, W&B for training metrics.
