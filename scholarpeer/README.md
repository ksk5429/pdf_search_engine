# ScholarPeer

> Citation-grounded AI co-reviewer. MARG-style multi-agent RAG over a locally-parsed scientific corpus.

Builds on the blueprint in [`../pdf_search_engine/A blueprint for an AI co-reviewer.md`](../pdf_search_engine/A%20blueprint%20for%20an%20AI%20co-reviewer.md).

## What it does

Given a paper (PDF or Markdown), ScholarPeer produces a structured peer review with every
claim grounded in retrieved passages from a local corpus + OpenAlex + Semantic Scholar.

```
Input paper
    v
Ingest (GROBID + MinerU 2.5-Pro)
    v
Index (Qdrant: dense + BM25 + ColPali)  +  LightRAG entity graph
    v
Retrieve (hybrid + visual + citation-graph)
    v
Reason (Claude Opus leader + 5 Sonnet specialists)
    |    - novelty   - methodology   - clarity
    |    - reproducibility   - related-work
    v
Synthesize (OpenScholar-style self-feedback, Haiku formatting)
    v
Structured review with [SP:xxx] / [OA:W...] / [S2:...] citations
```

## Quick start

### 1. Infrastructure (Docker)

```bash
cd scholarpeer
docker compose up -d qdrant grobid
# Qdrant -> http://localhost:6333 (dashboard :6333/dashboard)
# GROBID -> http://localhost:8070
```

### 2. Python environment

```bash
# With GPU (CUDA 12.1+)
pip install -e ".[mineru,colpali,lightrag,dev]"

# Install MinerU model weights (first run only)
magic-pdf --install-models
```

### 3. Configure

```bash
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY, POLITE_EMAIL, SP_CORPUS_DIR, SP_PDF_DIR
```

### 4. Run

```bash
# Ingest existing PDF corpus (one-time)
scholarpeer ingest --pdf-dir "F:/TREE_OF_THOUGHT/PHD/papers/new_pdf" \
                   --out-dir data/corpus

# Build indices (one-time per corpus)
scholarpeer index --corpus data/corpus --collection sp_dense

# Query the corpus
scholarpeer search "suction caisson lateral capacity sand" --top-k 10

# Full co-review of a target paper
scholarpeer review "path/to/target.pdf" --output review.md
```

## Phases implemented

| Phase | Layer | Status |
|-------|-------|--------|
| P1 | Ingest (GROBID + MinerU) | working |
| P2 | Index (Qdrant dense + BM25 hybrid) | working |
| P3 | Agents (Leader + 5 specialists) | working |
| P4 | Visual retrieval (ColPali) | working (requires GPU + byaldi) |
| P5 | LightRAG graph + OpenScholar self-feedback | working (optional layer) |

## Benchmarks targeted

- **ScholarQABench** — literature synthesis correctness & citation accuracy.
- **ViDoRe V3** — visual document retrieval (26k pages, 3,099 queries, 6 langs).
- **OmniDocBench v1.6** — PDF parsing quality (MinerU 2.5-Pro scored 95.69).

## Configuration reference

| Env var | Default | Meaning |
|---------|---------|---------|
| `ANTHROPIC_API_KEY` | (required) | For Claude Agent SDK |
| `POLITE_EMAIL` | (required) | For OpenAlex / Unpaywall / CrossRef polite pool |
| `SP_CORPUS_DIR` | `F:/TREE_OF_THOUGHT/PHD/papers/literature_review` | Parsed Markdown corpus |
| `SP_PDF_DIR` | `F:/TREE_OF_THOUGHT/PHD/papers/new_pdf` | Source PDFs |
| `SP_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `SP_GROBID_URL` | `http://localhost:8070` | GROBID endpoint |
| `SP_DENSE_MODEL` | `BAAI/bge-m3` | Dense embedding model (1024-dim) |
| `SP_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `SP_DEVICE` | `cuda` | `cuda` / `cpu` / `mps` |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed dataflow.

## License

Apache-2.0
