# BioDreamer Blog

Research updates, technical deep-dives, and project milestones.

Posts are written in Markdown and are readable both here on GitHub and on the [BioDreamer web application](#).

---

## Posts

| Date | Post | Topic |
|---|---|---|
| 2026-03-19 | [Introducing BioDreamer](posts/2026-03-19-introducing-biodreamer.md) | Vision, motivation, and roadmap |

---

## Writing a New Post

1. Create a new `.md` file in `blog/posts/` with the naming convention: `YYYY-MM-DD-slug.md`
2. Add YAML frontmatter at the top (see template below)
3. Write content in standard Markdown (GitHub-flavoured)
4. Add an entry to the table above
5. The web app automatically picks up new posts via the API

### Frontmatter Template

```yaml
---
title: "Your Post Title"
date: 2026-03-19
author: "Your Name"
tags: [protein-dreamer, world-models, active-inference]
summary: "One-sentence summary shown in the blog index."
---
```

### Supported Features

- Standard Markdown (headings, lists, code blocks, tables)
- LaTeX math via `$...$` (inline) and `$$...$$` (block)
- Images: place in `blog/assets/` and reference as `![alt](assets/image.png)`
- Code syntax highlighting (Python, TypeScript, YAML, bash)
- Internal links to other posts
