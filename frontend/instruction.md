# Frontend Development Instruction for BioDreamer Project Website

## Preparation

Before writing any code, read and understand the following files thoroughly:

- `README.md` — for the project's purpose, features, and technical overview
- `plans/project_overview.md` — for the broader vision and roadmap

---

## Pages & Content

Build a multi-page (or single-page with smooth scroll navigation) static website with the following sections:

### 1. Home

- Hero section with the project name, a concise tagline, and a short explanation of what BioDreamer does and why it matters
- Call-to-action buttons linking to the GitHub repo and documentation (ReadTheDocs — placeholder URL for now)
- Optional: a brief animated or visual element that hints at the biological/ML nature of the project (e.g. a subtle DNA strand, graph, or generative visual)

### 2. Quick Start

- Instructions on how to install the Python package (e.g. `pip install bio-dreamer`)
- A minimal code snippet showing the most basic usage
- Link to full documentation
- Note clearly that this section will be expanded once the project matures

### 3. Features

- A visual grid or card layout highlighting the key capabilities of the project (drawn from README and project overview)
- Icons or simple illustrations per feature
- Helps visitors immediately understand what the tool offers without reading docs

### 4. How It Works

- A high-level pipeline or architecture diagram (can be a simple illustrated flow: Input → Model → Output)
- 3–5 steps explaining the core workflow in plain language
- Bridges the gap between the hero pitch and the technical docs

### 5. Blog

- List of published posts with title, date, short excerpt, and a "Read more" link
- Pull content from the `blog/` directory
- Paginated or filtered by tag/topic if there are multiple posts
- Clean readable layout optimized for long-form text

### 6. Examples *(placeholder)*

- A clearly marked "Coming Soon" section with a brief description of the kinds of examples that will appear (e.g. use cases, notebooks, demo outputs)
- Optional: embed or link to a Google Colab notebook if one exists

### 7. Roadmap

- A visual timeline or kanban-style board showing planned milestones, current status, and completed work
- Drawn from `plans/project_overview.md`
- Helps contributors and users understand where the project is headed

### 8. Contact

- A clean card or section layout with all links:
  - 🔗 Project GitHub: https://github.com/LoqmanSamani/bio_dreamer
  - 📧 Email: samaniloqman91@gmail.com
  - 📖 ReadTheDocs: *(placeholder — not yet created)*
  - 👤 Personal GitHub: https://github.com/LoqmanSamani
  - 💼 LinkedIn: https://www.linkedin.com/in/loghman-samani-8a5208199/
  - 🐦 X (Twitter): https://x.com/Loqman_Samani
- Optional: a simple contact form (static, using Formspree or similar — no backend needed)

---

## Design & Style

Before generating any code, **present the following choices** and wait for confirmation:

## Deployment

- Configure the site to deploy automatically to GitHub Pages
- If using a static site generator, add a GitHub Actions workflow (`.github/workflows/deploy.yml`) that builds and publishes on every push to `main`
- Include a note in `README.md` with the live URL once deployed
