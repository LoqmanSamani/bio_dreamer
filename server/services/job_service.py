"""
server.services.job_service — Job Queue Management Service.

Purpose:
    Manages the lifecycle of asynchronous jobs: creation, status tracking,
    cancellation, result retrieval, and cleanup.

Components to implement:
    - JobService:
        - create_job(job_type, params) → job_id (creates DB record + dispatches to worker)
        - get_status(job_id) → JobStatus
        - get_results(job_id) → job-type-specific results
        - cancel_job(job_id) → sends cancellation signal to worker
        - list_jobs(filters) → paginated job list
        - cleanup_old_jobs(max_age_days) → delete expired job records

Design notes:
    - Uses Redis (via rq or celery) as the message broker for dispatching
      tasks to background workers.
    - Job results are stored in the database (small metadata) and on disk
      (large files: trajectories, structures).
    - SSE (Server-Sent Events) endpoint for real-time progress streaming.
"""
