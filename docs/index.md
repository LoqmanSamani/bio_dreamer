# BioDreamer

**World Models for Biological Design**

BioDreamer applies model-based reinforcement learning with latent-space world models to biological design at three scales:

| Module | Scale | What It Does |
|---|---|---|
| **MolDreamer** | Atomic | Molecular dynamics in latent space |
| **ProteinDreamer** | Protein | Protein fitness landscape navigation via dreaming |
| **CellDreamer** | Cellular | Cell reprogramming perturbation planning |

## Quick Links

- [Architecture Overview](architecture.md) — how the JEPA world model and Active Inference policy work together.
- [Python Library Reference](api/core/index.md) — auto-generated docs for `biodreamer.*` modules.
- [REST API Reference](rest_api.md) — FastAPI endpoints for the web interface.
- [Deployment Guide](deployment.md) — Docker, GPU workers, and production setup.

## Installation

```bash
pip install -e ".[dev,docs]"
```

## Building these docs

```bash
make docs        # build static site to site/
make docs-serve  # live-reload dev server at http://localhost:8000
```
