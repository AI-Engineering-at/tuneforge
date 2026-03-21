# open-notebook Pipeline

## Purpose

TuneForge uses open-notebook as the private source of truth for research notes, benchmark interpretation, release decisions, and content ideation.

## Internal Record Structure

Each notebook entry should capture:

- goal
- hypothesis
- config
- hardware
- runtime and cost
- benchmark result
- release decision
- public-release level

## Public Export Rule

Only approved notebook entries may be turned into public assets.

Public assets include:

- wiki pages
- blog posts
- social snippets
- release notes
- validation-registry candidate entries after governance approval

Internal-only content stays private:

- raw notes
- unpublished experiments
- customer-sensitive observations
- strategy discussion

## Export Mechanism

Use:

```bash
python scripts/render_content_bundle.py <notebook-export.json> <output-dir>
```

Generated files:

- `wiki.md`
- `blog.md`
- `social.txt`
- `release-notes.md`
