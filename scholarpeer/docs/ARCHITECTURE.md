# ScholarPeer Architecture

5-layer system derived from the blueprint [`../../pdf_search_engine/A blueprint for an AI co-reviewer.md`](../../pdf_search_engine/A%20blueprint%20for%20an%20AI%20co-reviewer.md).

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Layer 5 — SYNTHESIZE                                                     │
 │   OpenScholar self-feedback loop  ->  Haiku Markdown formatter           │
 └────────────────────────────────▲─────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴─────────────────────────────────────────┐
 │ Layer 4 — REASON (Claude Agent SDK)                                      │
 │                                                                          │
 │   Opus Leader                                                            │
 │     ├─ plan retrieval queries                                            │
 │     ├─ fan out 5 Sonnet specialists (parallel by default):               │
 │     │     novelty | methodology | clarity | reproducibility | related_work │
 │     └─ aggregate grounded ReviewerComment[]                              │
 └────────────────────────────────▲─────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴─────────────────────────────────────────┐
 │ Layer 3 — RETRIEVE                                                       │
 │                                                                          │
 │   HybridRetriever ──► dense (BGE-M3) + sparse (BM25) + RRF + BGE-rerank │
 │   ColPaliRetriever ──► page-image late-interaction MAX_SIM               │
 │   OpenAlex / Semantic Scholar ──► external prior-art + citation graph    │
 │   LightRAG traverser ──► PPR multi-hop over entity-relation graph        │
 └────────────────────────────────▲─────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴─────────────────────────────────────────┐
 │ Layer 2 — INDEX (Qdrant, self-hosted via Docker)                         │
 │                                                                          │
 │   sp_dense       : dense cosine + BM25 sparse, binary quantization       │
 │   sp_colpali     : multi-vector MAX_SIM (ColQwen2.5)                     │
 │   lightrag.gpickle : NetworkX MultiDiGraph (entity-relation triples)     │
 └────────────────────────────────▲─────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴─────────────────────────────────────────┐
 │ Layer 1 — INGEST                                                         │
 │                                                                          │
 │   PDF ─► GROBID      (metadata + references)  +  MinerU 2.5-Pro (body)   │
 │   Markdown ─► MarkdownIngester (heuristic metadata)                      │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Data contracts

| Producer | Output | Consumer |
|----------|--------|----------|
| Ingest   | `Paper` (json) + raw Markdown + sections.json | Index, Agents |
| Index    | `sp_dense` / `sp_colpali` collections | Retrieve |
| Retrieve | `RetrievalHit` + `RetrievalLog` | Agents, Grounding |
| Agents   | `Review` with `ReviewerComment[]` | Synthesize, Grounding |
| Synthesize | final Markdown review | user |

All schemas live in [`src/scholarpeer/schemas/`](../src/scholarpeer/schemas). Schemas are Pydantic v2, frozen where possible.

## Citation grounding (hard invariant)

Every `[SOURCE:id]` in the generated review is verified against the session's
`RetrievalLog` by [`scholarpeer.eval.citation_grounding.verify_grounding`](../src/scholarpeer/eval/citation_grounding.py).
If any citation is ungrounded, the run is flagged and the review can be regenerated.

This backstops the MARG/OpenScholar-style agent instructions that already require
citation-only assertions. Without this check the 78–90% hallucination rate
documented in the blueprint would creep back in.

## Model routing

| Role | Default model | Rationale |
|------|---------------|-----------|
| Leader / planner | `claude-opus-4-7` | Long-horizon review strategy |
| Specialists | `claude-sonnet-4-6` | Fast per-section analysis |
| Formatter / graph extractor | `claude-haiku-4-5` | Cheap polish + triple extraction |

Override via `SP_LEADER_MODEL`, `SP_SPECIALIST_MODEL`, `SP_FORMATTER_MODEL`.

## Failure modes we've thought about

1. **MinerU unavailable** — falls back to `pymupdf4llm` (CPU-only), quality drops
   but pipeline continues.
2. **GROBID unavailable** — metadata is reconstructed from filename heuristics.
3. **Qdrant connection lost** — indexing fails loudly; retrieval raises a clear
   `httpx` error rather than returning empty results silently.
4. **A specialist crashes** — other specialists continue (per-future try/except in
   `LeaderAgent._dispatch_specialists`).
5. **Context rot during self-feedback** — bounded by `self_feedback_rounds`
   (default 2) and per-specialist focus excerpts.
