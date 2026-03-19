"""
blog_service.py — Blog Post Service.

Purpose:
    Business logic for reading, parsing, caching, and searching blog posts.
    Called by the blog router. Decouples file I/O and parsing from the
    HTTP layer.

Classes / Functions to implement:

    @dataclass
    class BlogPost:
        slug: str               # filename without .md (e.g. "2026-03-19-introducing-biodreamer")
        title: str              # from YAML frontmatter
        date: datetime.date     # from YAML frontmatter
        author: str             # from YAML frontmatter
        tags: list[str]         # from YAML frontmatter
        summary: str            # from YAML frontmatter
        content_md: str         # raw Markdown body (after frontmatter)
        content_html: str       # rendered HTML
        file_mtime: float       # last modification time (for cache invalidation)

    class BlogService:
        posts_dir: Path         # path to blog/posts/

        def list_posts(tag?, search?, page?, limit?) → list[BlogPost]:
            Scan posts_dir for .md files, parse frontmatter, return
            sorted by date descending. Supports tag filter and keyword
            search (matches title + summary + tags).

        def get_post(slug: str) → BlogPost | None:
            Read and parse a single post by slug. Returns None if not found.

        def _parse_post(path: Path) → BlogPost:
            Split file at --- delimiters, parse YAML header, render
            Markdown body to HTML.

        def _render_markdown(md: str) → str:
            Convert Markdown to HTML with syntax highlighting, math,
            heading IDs, and safe link handling.

        def refresh_cache() → None:
            Re-scan posts_dir and update in-memory cache. Called on
            startup and optionally via admin endpoint.

    Caching:
        - In-memory dict[slug → BlogPost]
        - On get_post(): check file mtime; if changed, re-parse
        - Thread-safe (asyncio-compatible)
"""
