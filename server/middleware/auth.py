"""
server.middleware.auth — Authentication Middleware.

Purpose:
    Optional authentication layer for the API. Disabled in local development,
    enabled in production to protect compute-heavy endpoints.

Components to implement:
    - APIKeyMiddleware:
        Validates API key from Authorization header or query parameter.
        Skips authentication for /health and /docs endpoints.

    - RateLimitMiddleware:
        Limits request rate per client IP to prevent abuse of GPU resources.
        Configurable limits per endpoint (e.g., /submit has lower limits than /results).

Design notes:
    - Authentication is opt-in (controlled by AUTH_ENABLED in .env).
    - For production deployment, consider adding OAuth2 / JWT support.
    - Rate limiting uses in-memory token bucket (dev) or Redis (production).
"""
