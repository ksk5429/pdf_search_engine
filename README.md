# PDF Search Engine

Multi-source academic PDF search and download engine with PDF-to-Markdown conversion. Discovers open-access papers from OpenAlex, Semantic Scholar, CrossRef, and Unpaywall.

## How It Works

```
Research Topics (10 configurable)
    |
    v
┌──────────┐  ┌──────────────────┐  ┌──────────┐
│ OpenAlex  │  │ Semantic Scholar │  │ CrossRef │
│ (primary) │  │ (citation graph) │  │ (DOIs)   │
└─────┬────┘  └────────┬─────────┘  └────┬─────┘
      └────────────────┼──────────────────┘
                       v
              Paper metadata pool
              (doi, title, year, authors, cited_by, pdf_url)
                       |
              Deduplication (DOI + MD5 title hash)
                       |
              Relevance filter (domain keywords)
                       |
              Unpaywall fallback (DOI -> OA PDF)
                       |
              Download + validate (magic bytes, size check)
                       |
              *.pdf -> pymupdf4llm -> *.md (with quality validation)
```

## Quick Start

```bash
pip install requests

# Set your email (required by APIs for polite access)
export POLITE_EMAIL=your@email.com

# Full search (all 10 topics)
python pdf_search_engine.py

# Test mode (2 topics, 5 results each)
python pdf_search_engine.py --test

# Single topic
python pdf_search_engine.py --topic "scour"

# Custom query
python pdf_search_engine.py --query "suction caisson lateral capacity sand"

# Skip Semantic Scholar (if rate-limited)
python pdf_search_engine.py --no-s2

# Custom directories
python pdf_search_engine.py --download-dir ./pdfs --output-dir ./markdown

# Convert downloaded PDFs to Markdown
pip install pymupdf4llm
python convert_to_markdown.py
python convert_to_markdown.py --dry-run      # preview only
python convert_to_markdown.py --recheck      # re-validate existing
```

## Features

- **4 API sources**: OpenAlex (primary), Semantic Scholar, CrossRef, Unpaywall
- **Smart deduplication**: DOI + MD5 title hash, persisted across runs in `known_papers.json`
- **Citation sorting**: Downloads highest-cited papers first
- **Relevance filtering**: 70+ domain keywords checked against titles
- **PDF validation**: Magic byte check on first chunk + minimum 10KB size
- **Streaming downloads**: Chunked transfer, single-pass validation (no double-fetch)
- **Crash-resilient**: State saved after each topic (resume where you left off)
- **Markdown quality validation**: Detects garbled text, scanned PDFs, empty extractions
- **Fuzzy filename matching**: Avoids re-converting papers across naming conventions

## Research Topics

Ships with 10 pre-configured topics (edit `RESEARCH_TOPICS` in source):

| # | Topic | Example Queries |
|---|-------|----------------|
| 1 | Suction bucket foundations | suction caisson capacity, tripod cyclic loading |
| 2 | Scour detection | scour monitoring, natural frequency vibration |
| 3 | Structural health monitoring | SHM offshore wind, ambient vibration |
| 4 | Soil-structure interaction | p-y curves, Winkler models, dynamic impedance |
| 5 | Geotechnical reliability | Monte Carlo, LRFD, probabilistic capacity |
| 6 | Centrifuge modelling | physical modelling, similitude |
| 7 | Dimensional analysis | Buckingham Pi, scaling laws |
| 8 | Numerical modelling | FEM, OptumG2, OpenSeesPy, OpenFAST |
| 9 | Machine learning in geotechnics | autoencoders, transfer learning |
| 10 | Offshore wind design standards | DNV, IEC 61400, lifetime assessment |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POLITE_EMAIL` | (required) | Your email for API polite pools |
| `PDF_DOWNLOAD_DIR` | `./downloads` | Where PDFs are saved |
| `PDF_OUTPUT_DIR` | `./markdown` | Where markdown files are saved |

## API Rate Limits

| API | Rate Limit | Handling |
|-----|-----------|----------|
| OpenAlex | Polite pool (unlimited with email) | 1s delay |
| Semantic Scholar | ~100/5min | Exponential backoff (10s/20s/40s) |
| CrossRef | Polite pool (unlimited with email) | 1s delay |
| Unpaywall | 100K/day (with email) | 0.5s delay |

## Architecture

```
pdf_search_engine/
├── pdf_search_engine.py    # Multi-source search + download
├── convert_to_markdown.py  # PDF-to-Markdown with quality validation
├── pyproject.toml
└── README.md

Generated at runtime:
├── downloads/              # Downloaded PDFs
├── markdown/               # Converted markdown files
├── known_papers.json       # Deduplication store
├── download_log.json       # Audit trail
└── search_engine.log       # Session log
```

## Ecosystem

| Repo | Purpose |
|------|---------|
| [ai_style_checker](https://github.com/ksk5429/ai_style_checker) | 12-checker AI detection + fingerprinting |
| [sentence_evolver](https://github.com/ksk5429/sentence_evolver) | 10-persona sentence rewriting + A/B scoring |
| [publishing_engine](https://github.com/ksk5429/publishing_engine) | .qmd to publication DOCX (7 document types) |
| [manuscript_pipeline](https://github.com/ksk5429/manuscript_pipeline) | Orchestrator chaining all engines |
| **pdf_search_engine** | Academic PDF discovery + conversion (this repo) |

## License

Apache 2.0
