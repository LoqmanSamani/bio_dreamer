"""
blog.py — Pydantic Schemas for Blog API.

Purpose:
    Request/response models for the blog endpoints.

Schemas to define:

    class BlogPostSummary(BaseModel):
        slug: str               # URL-safe identifier
        title: str
        date: date
        author: str
        tags: list[str]
        summary: str

    class BlogPostDetail(BlogPostSummary):
        content: str            # Rendered HTML of the Markdown body

    class BlogPostListResponse(BaseModel):
        posts: list[BlogPostSummary]
        total: int
        page: int
        limit: int
"""
