# Deployment

## Prerequisites

- Docker Desktop (for Qdrant + GROBID)
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (required for MinerU + ColPali; optional for BGE-M3)
- ~30 GB free disk for model weights + corpus index

## Setup

```bash
cd F:/TREE_OF_THOUGHT/scholarpeer
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[mineru,colpali,lightrag,dev]"
magic-pdf --install-models       # one-time (~10 GB)
```

## Services

```bash
docker compose up -d qdrant grobid
scholarpeer status                # confirm reachability
```

Endpoints:
- Qdrant: http://localhost:6333 (dashboard: `/dashboard`)
- GROBID: http://localhost:8070

## First-time corpus build

```bash
# 1) Parse the existing papers (PDF + MD) into data/corpus/
python scripts/ingest_corpus.py

# 2) Build the Qdrant hybrid index
python scripts/build_index.py

# 3) (Optional, GPU only) build the ColPali visual index
scholarpeer index-visual

# 4) (Optional) build the LightRAG entity-relation graph
scholarpeer graph-build --max-chunks 15
```

## Smoke test

```bash
pytest -m unit                    # unit tests — no services required
scholarpeer search "suction caisson lateral capacity sand" --top-k 5
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `magic_pdf` ImportError at ingest | Install MinerU: `pip install 'magic-pdf[full]'` |
| Qdrant 404 on collection | Run `scholarpeer status` then `python scripts/setup_qdrant.py` |
| BGE-M3 download stuck | Set `HF_HUB_ENABLE_HF_TRANSFER=1` and retry |
| CUDA OOM during rerank | Drop `SP_TOP_K_RERANK` or set `SP_DEVICE=cpu` for reranker only |
| Citation grounding warns on every run | Enable verbose logs (`SP_LOG_LEVEL=DEBUG`) and inspect `RetrievalLog.hits` |
