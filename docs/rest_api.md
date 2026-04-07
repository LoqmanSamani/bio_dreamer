# REST API Reference

The BioDreamer backend is a FastAPI application. When running, interactive docs
are available at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Below is a summary of endpoint groups.  The canonical, always-up-to-date
reference is the auto-generated OpenAPI schema served by FastAPI at
`/openapi.json`.

---

## Protein Dreamer — `/api/protein-dreamer/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Submit a protein design job (wild-type sequence + objectives) |
| `GET`  | `/results` | Retrieve dream trajectories, ranked candidates, fitness plots |
| `POST` | `/evaluate` | Score a candidate against a structure oracle |

## Mol Dreamer — `/api/mol-dreamer/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Submit a molecular dynamics world-model job |
| `GET`  | `/results` | Retrieve latent trajectories and property predictions |

## Cell Dreamer — `/api/cell-dreamer/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Submit a perturbation planning job |
| `GET`  | `/results` | Retrieve predicted cell state trajectories |

## Models — `/api/models/`

| Method | Path | Description |
|---|---|---|
| `GET`  | `/list` | List available models (local + Hugging Face Hub) |
| `POST` | `/load` | Load a model into GPU memory |
| `POST` | `/upload` | Push a trained model to Hugging Face Hub |

## Jobs — `/api/jobs/`

| Method | Path | Description |
|---|---|---|
| `GET`  | `/` | List all jobs for the current session |
| `GET`  | `/{job_id}` | Get status and partial results of a specific job |
| `POST` | `/{job_id}/cancel` | Cancel a running job |

## Health — `/health`

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe for Docker / Kubernetes |
