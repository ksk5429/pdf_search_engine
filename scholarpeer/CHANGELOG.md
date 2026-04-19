# Changelog

## 0.1.0 — 2026-04-19

Initial scaffold following the blueprint
"A blueprint for an AI co-reviewer" (April 2026).

### Added — 5-layer architecture

- **Layer 1 — Ingest** (`scholarpeer.ingest`)
  - `GrobidClient` for metadata + references via local GROBID service
  - `MinerUParser` — MinerU 2.5-Pro primary, pymupdf4llm fallback
  - `MarkdownIngester` — reuse existing `literature_review/*.md` corpus
  - `IngestPipeline` — dual-backend orchestration, writes paired `.md` + `.json`

- **Layer 2 — Index** (`scholarpeer.index`)
  - `SectionChunker` — token-aware, section-respecting, tiktoken-measured
  - `DenseEmbedder` — BGE-M3 (1024-dim, multilingual)
  - `SparseEmbedder` — FastEmbed BM25 via Qdrant
  - `QdrantStore` — hybrid + ColPali collections, binary quantization, `sp_` prefix guard
  - `ColPaliIndexer` — ColQwen2.5 multi-vector late-interaction

- **Layer 3 — Retrieve** (`scholarpeer.retrieve`)
  - `HybridRetriever` — dense + sparse + RRF + cross-encoder rerank
  - `ColPaliRetriever` — byaldi-backed visual retrieval
  - `OpenAlexClient` / `SemanticScholarClient` — external citation graph
  - `reciprocal_rank_fusion` — score-free RRF (k=60 default)

- **Layer 4 — Reason** (`scholarpeer.agents`)
  - `LeaderAgent` (Opus) — plans retrieval, dispatches specialists, aggregates
  - Five MARG-style specialists (Sonnet):
    `novelty`, `methodology`, `clarity`, `reproducibility`, `related_work`
  - Parallel dispatch via `concurrent.futures.ThreadPoolExecutor`

- **Layer 5 — Synthesize** (`scholarpeer.synthesize`)
  - `SelfFeedbackLoop` — OpenScholar-style critic + targeted retrieval
  - `ReviewFormatter` (Haiku) — final Markdown polish, citation-preserving

- **LightRAG graph layer** (`scholarpeer.graph`)
  - `EntityRelationExtractor` — Haiku-based triple extraction (~$0.15/doc)
  - `GraphStore` — NetworkX MultiDiGraph, pickle-persisted
  - `GraphTraverser` — personalized PageRank + multi-hop BFS

- **Evaluation** (`scholarpeer.eval`)
  - `verify_grounding` — hard check that every `[SOURCE:id]` exists in `RetrievalLog`

- **Infrastructure**
  - `docker-compose.yml` — self-hosted Qdrant 1.12 + GROBID 0.8
  - `CLAUDE.md` — project conventions (<300 lines, progressive disclosure)
  - Pydantic v2 schemas with frozen models where applicable
  - Typer CLI (`scholarpeer {ingest,index,search,review,graph-build,status}`)
  - 6 unit test files (schemas, fusion, chunker, grounding, hashing, markdown ingest)
  - 1 integration test (Qdrant round-trip)

### Notes

- BGE-M3 + ColPali require CUDA GPU for realistic throughput.
- MinerU weights must be installed separately via `magic-pdf --install-models`.
- `POLITE_EMAIL` is required for OpenAlex / Unpaywall polite-pool access.
