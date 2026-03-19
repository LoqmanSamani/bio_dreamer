"""
blog.py — Blog API Router.

Purpose:
    REST endpoints for serving blog posts. Reads Markdown files from
    the blog/posts/ directory, parses YAML frontmatter, and returns
    post metadata and rendered HTML content.

    Posts live as .md files in the repository so they are readable
    directly on GitHub. This router makes the same content available
    through the web application's API.

Endpoints:
    GET /api/blog/posts
        List all posts in reverse chronological order.
        Response: Array of { slug, title, date, author, tags, summary }
        Query params: ?tag=<tag> (filter), ?search=<keyword>, ?page=1&limit=10

    GET /api/blog/posts/{slug}
        Retrieve a single post by slug.
        Response: { slug, title, date, author, tags, summary, content }
        content is Markdown rendered to HTML (via markdown-it or similar).

Implementation notes:
    - Reads blog/posts/*.md from disk (relative to project root)
    - Parses YAML frontmatter with pyyaml (between --- delimiters)
    - Renders Markdown body to HTML with:
        • Fenced code block syntax highlighting (Pygments)
        • LaTeX math support ($ and $$)
        • Heading IDs for anchor linking
    - Caches parsed posts in memory (dict[slug] → ParsedPost)
    - Cache is invalidated on file mtime change (dev mode) or on restart

Dependencies:
    - pyyaml (frontmatter parsing)
    - markdown or markdown-it-py (Markdown → HTML)
    - pygments (code highlighting)
"""
