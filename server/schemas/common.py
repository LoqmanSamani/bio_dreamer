"""
server.schemas.common — Shared Pydantic Models.

Purpose:
    Common request/response schemas used across all API routers:
    job status, pagination, error responses, file upload metadata.

Components to implement:
    - JobStatus: id, status, type, created_at, updated_at, progress, result_url
    - PaginationParams: page, page_size, sort_by, sort_order
    - ErrorResponse: error_code, message, details
    - FileUploadMeta: filename, content_type, size_bytes
    - ModelReference: model_id, source, version
"""
