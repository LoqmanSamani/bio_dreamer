/**
 * BlogRenderer — Markdown Content Renderer.
 *
 * Purpose:
 *   Renders blog post Markdown content as styled React components.
 *   Handles code highlighting, math, images, and heading anchors.
 *
 * Props:
 *   - content: string         — raw Markdown string or pre-rendered HTML
 *   - format?: "markdown" | "html"  — input format (default: "html")
 *
 * Features:
 *   - If format="markdown": uses react-markdown with remark/rehype plugins
 *   - If format="html": uses dangerouslySetInnerHTML with DOMPurify sanitisation
 *   - Syntax highlighting for code blocks (Python, TypeScript, YAML, bash)
 *   - KaTeX rendering for $...$ and $$...$$ math blocks
 *   - Responsive images with lazy loading
 *   - Heading IDs for table-of-contents linking
 *   - Styled blockquotes, tables, and horizontal rules
 *   - Tailwind Typography plugin (`prose`) for consistent styling
 *
 * Security:
 *   - HTML content is sanitised with DOMPurify before rendering
 *   - No script execution from post content
 */
