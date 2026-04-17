# Changelog

## 2026.04.1 (2026-04-17)

### Added
- Initial release: multi-source academic PDF search engine
- 4 API sources: OpenAlex, Semantic Scholar, CrossRef, Unpaywall
- Smart deduplication (DOI + MD5 title hash)
- Citation-sorted downloads, relevance filtering
- PDF validation (magic bytes, streaming, single-pass)
- PDF-to-Markdown conversion via pymupdf4llm
- Quality validation (garbled text, scanned PDF detection)
- 10 pre-configured research topics
- Configurable paths via environment variables and CLI args
