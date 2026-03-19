/**
 * Blog Post Page — /blog/[slug]
 *
 * Purpose:
 *   Renders a single blog post. The slug matches the filename in
 *   blog/posts/ (e.g., "2026-03-19-introducing-biodreamer").
 *
 * Layout:
 *   - Post header: title, date, author, tags
 *   - Rendered Markdown content with:
 *       • Syntax-highlighted code blocks (Prism or Shiki)
 *       • LaTeX math rendering (KaTeX)
 *       • Responsive images
 *       • Anchor links on headings
 *   - Table of contents sidebar (auto-generated from headings)
 *   - Previous / Next post navigation at the bottom
 *   - "Edit on GitHub" link → direct to the .md source file in the repo
 *
 * Data fetching:
 *   - GET /api/blog/posts/[slug] → { title, date, author, tags, content (HTML) }
 *   - generateStaticParams() fetches all slugs for SSG
 *
 * Dependencies:
 *   - react-markdown or next-mdx-remote for Markdown → React rendering
 *   - rehype-highlight / rehype-katex for code + math
 *   - rehype-slug + rehype-autolink-headings for anchor links
 */
