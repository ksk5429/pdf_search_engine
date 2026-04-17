# PDF Search Engine

Multi-source academic PDF search and download engine with PDF-to-Markdown conversion. Discovers open-access papers from OpenAlex, Semantic Scholar, CrossRef, and Unpaywall.

## How It Works

```
Research Topics (configurable)
    |
    v
OpenAlex --> Semantic Scholar --> CrossRef
    |              |                  |
    +--------------+------------------+
                   |
         Paper metadata pool
         (doi, title, year, authors, pdf_url, cited_by)
                   |
         Deduplication (DOI + title hash)
                   |
         Relevance filter (domain keywords)
                   |
         Unpaywall fallback (DOI -> OA PDF)
                   |
         Download + validate (magic bytes, size)
                   |
         PDF-to-Markdown (pymupdf4llm)
                   |
         Quality validation (garbled text, length)
```

## Quick Start

```bash
pip install requests

# Set your email (required by APIs for polite access)
export POLITE_EMAIL=your@email.com

# Search and download (all topics)
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
```

## PDF-to-Markdown Conversion

```bash
pip install pymupdf4llm

# Convert all downloaded PDFs
python convert_to_markdown.py

# Preview what would be converted
python convert_to_markdown.py --dry-run

# Convert a single file
python convert_to_markdown.py --single "(2023 Byrne) Suction caissons.pdf"

# Re-validate existing conversions
python convert_to_markdown.py --recheck
```

## Features

- **4 API sources**: OpenAlex (primary), Semantic Scholar, CrossRef, Unpaywall
- **Smart deduplication**: DOI + MD5 title hash, persisted across runs
- **Relevance filtering**: Domain keyword matching in titles
- **Citation sorting**: Downloads highest-cited papers first
- **PDF validation**: Magic byte check + minimum size (10KB)
- **Crash-resilient**: State saved after each topic (resume where you left off)
- **Markdown quality validation**: Detects garbled text, scanned PDFs, empty extractions
- **Fuzzy filename matching**: Avoids re-converting papers across naming conventions

## Configuration

All paths configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `POLITE_EMAIL` | (required) | Your email for API polite pools |
| `PDF_DOWNLOAD_DIR` | `./downloads` | Where PDFs are saved |
| `PDF_OUTPUT_DIR` | `./markdown` | Where markdown files are saved |

## Research Topics

The engine ships with 10 pre-configured research topics (edit `RESEARCH_TOPICS` in `pdf_search_engine.py`):

1. Suction bucket foundations
2. Scour detection and monitoring
3. Structural health monitoring (SHM)
4. Soil-structure interaction (SSI)
5. Geotechnical reliability
6. Centrifuge modelling
7. Dimensional analysis
8. Numerical modelling (FEM, OpenSees, OpenFAST)
9. Machine learning in geotechnics
10. Offshore wind design standards

## Architecture

```
pdf_search_engine/
├── pdf_search_engine.py    # Multi-source search + download engine
├── convert_to_markdown.py  # PDF-to-Markdown with quality validation
├── pyproject.toml
└── README.md
```

## API Rate Limits

| API | Rate Limit | Handling |
|-----|-----------|----------|
| OpenAlex | Polite pool (unlimited with email) | 1s delay between calls |
| Semantic Scholar | ~100/5min | Exponential backoff (10s/20s/40s) |
| CrossRef | Polite pool (unlimited with email) | 1s delay |
| Unpaywall | 100K/day (with email) | 0.5s delay |

## Related Projects

- [ai_style_checker](https://github.com/ksk5429/ai_style_checker) -- Detect AI writing patterns
- [sentence_evolver](https://github.com/ksk5429/sentence_evolver) -- Multi-agent sentence rewriting
- [publishing_engine](https://github.com/ksk5429/publishing_engine) -- DOCX rendering for journals
- [manuscript_pipeline](https://github.com/ksk5429/manuscript_pipeline) -- Pipeline orchestrator

## License

Apache 2.0
