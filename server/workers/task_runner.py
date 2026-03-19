"""
server.workers.task_runner — Background Task Execution Engine.

Purpose:
    Runs compute-intensive tasks (dreaming, training, model upload) in a
    separate worker process with GPU access. Communicates with the API
    server via Redis job queue.

Components to implement:
    - TaskRunner:
        - run_dream_job(job_params) → InferenceResult (dispatched from dream endpoints)
        - run_training_job(job_params) → training metrics (dispatched from training endpoints)
        - run_upload_job(job_params) → HF Hub URL (dispatched from model upload)
        - run_active_learning_round(job_params) → new candidates + updated model

    - Progress reporting:
        Updates job progress (0-100%) in the database at regular intervals.
        The API server streams these updates to the frontend via SSE.

Design notes:
    - Worker process is separate from the API server process — they communicate
      only through Redis and the database. This allows the worker to run on a
      GPU node while the API server runs on a CPU node.
    - Uses rq (Redis Queue) or Celery for task dispatch and lifecycle management.
    - GPU memory is managed per-worker: one model loaded at a time, freed after
      each job (or kept warm for repeated jobs of the same type).
    - Graceful shutdown: finish current job before exiting on SIGTERM.
"""
