"""
server.config — Server Configuration.

Purpose:
    Loads and validates server configuration from configs/server.yaml and
    environment variables (.env). Provides a typed Settings object.

Components to implement:
    - Settings (pydantic BaseSettings):
        - server_host, server_port, cors_origins
        - database_url
        - redis_url (for job queue)
        - model_cache_dir, max_loaded_models
        - hf_token, hf_org
        - log_level
        - auth_enabled, api_key (for production)

Design notes:
    - Uses pydantic-settings for automatic env var parsing and validation.
    - .env file is loaded automatically (dotenv support).
    - Sensitive values (HF_TOKEN, API_KEY) are marked as SecretStr.
"""
