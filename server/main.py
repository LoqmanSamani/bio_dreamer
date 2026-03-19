"""
server.main — FastAPI Application Entry Point.

Purpose:
    Creates and configures the FastAPI application instance. Registers all
    routers, middleware, startup/shutdown events, and exception handlers.

Components to implement:
    - create_app() → FastAPI instance
        - Mount routers: /api/protein-dreamer, /api/mol-dreamer, /api/cell-dreamer,
          /api/models, /api/jobs
        - Add middleware: CORS (for frontend), request logging, rate limiting
        - Startup events: load default models, initialise job queue connection
        - Shutdown events: gracefully stop workers, save state
        - Exception handlers: structured error responses (JSON)

    - CLI entry point: uvicorn server.main:app --host 0.0.0.0 --port 8000

Design notes:
    - The server is stateless between requests — all job state is in the database.
    - Model loading is lazy (loaded on first request, cached in memory).
    - CORS origins are configured via SERVER.YAML / .env to allow frontend access.
    - Health check endpoint at /health for Docker / K8s probes.
"""
