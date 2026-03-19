"""
biodreamer.utils.logging — Structured Logging Configuration.

Purpose:
    Centralised logging setup for the entire BioDreamer project. Configures
    Python's logging module with structured output for console, file, and
    optionally Weights & Biases.

Components to implement:
    - setup_logging(level, log_file, wandb_project) → configured logger
    - get_logger(name) → module-specific logger
    - log_config(config_dict) → pretty-print and log training configuration
    - log_metrics(metrics_dict, step) → log metrics to all sinks (console + wandb)

Design notes:
    - Use structlog or Python's built-in logging with JSON formatter for
      structured log output (easy to parse in production).
    - Log levels: DEBUG for development, INFO for training runs, WARNING for production server.
    - The web server uses a separate logger configuration (server/config.py).
"""
