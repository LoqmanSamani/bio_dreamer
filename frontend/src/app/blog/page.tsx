/**
 * Blog Index Page — /blog
 *
 * Purpose:
 *   Lists all blog posts in reverse chronological order. Posts are
 *   authored as Markdown files in blog/posts/ and served via the API.
 *
 * Layout:
 *   - Page title: "Blog" with subtitle "Research updates and project milestones"
 *   - Filter/search bar (by tag, keyword)
 *   - Post cards (one per post), each showing:
 *       • Title (links to /blog/[slug])
 *       • Date, author
 *       • Tags as coloured badges
 *       • Summary (1-2 sentences from frontmatter)
 *   - Pagination if > 10 posts
 *
 * Data fetching:
 *   - GET /api/blog/posts → list of { slug, title, date, author, tags, summary }
 *   - Server-side rendered (SSR) or statically generated (SSG) for SEO
 */
