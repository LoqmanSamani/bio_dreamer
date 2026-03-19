"""
test_api.py — Integration Tests for FastAPI Server.

Purpose:
    Tests the REST API endpoints using httpx AsyncClient against the
    FastAPI app. Covers request validation, job submission, status
    polling, and error handling.

Tests to implement:
    - test_health_endpoint: GET /api/health returns 200 with status "ok"
    - test_protein_dreamer_submit: POST valid request → 202 with job_id
    - test_protein_dreamer_submit_invalid: missing sequence → 422
    - test_mol_dreamer_submit: POST valid simulation request → 202
    - test_cell_dreamer_submit: POST valid perturbation request → 202
    - test_job_status: GET /api/jobs/{id} returns current status
    - test_job_cancel: POST /api/jobs/{id}/cancel returns 200
    - test_model_list: GET /api/models returns model catalogue
    - test_model_load_unload: POST load then unload cycle
    - test_rate_limiting: excessive requests get 429 (if enabled)
    - test_cors_headers: preflight OPTIONS returns correct CORS headers
"""
