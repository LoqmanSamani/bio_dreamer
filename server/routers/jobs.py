"""
server.routers.jobs — Job Lifecycle Management API.

Purpose:
    REST API for tracking asynchronous jobs (training runs, inference tasks,
    model uploads). The frontend's "Job Tracker" page polls these endpoints.

Endpoints to implement:
    GET /api/jobs/
        Response: list of all jobs (recent first) with status, type, timestamps

    GET /api/jobs/{job_id}
        Response: detailed job status (progress %, partial results, logs)

    POST /api/jobs/{job_id}/cancel
        Response: cancellation acknowledgement

    DELETE /api/jobs/{job_id}
        Response: remove job record (only completed/failed jobs)

    GET /api/jobs/{job_id}/logs
        Response: streaming job logs (Server-Sent Events)

Design notes:
    - Jobs are persisted in the database (SQLite for dev, PostgreSQL for prod).
    - Job types: 'dream' (inference), 'train' (training), 'upload' (HF push),
      'download' (HF pull), 'active_learning' (AL round).
    - Status: 'queued', 'running', 'completed', 'failed', 'cancelled'.
    - Long-running jobs dispatch work to server/workers/task_runner.py via Redis.
"""
