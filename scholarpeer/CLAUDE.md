# ScholarPeer — CLAUDE.md

> Progressive disclosure: this file holds only universally-applicable rules. Task-specific
> guidance lives in `docs/`. Keep this file under 300 lines.

## What this is

ScholarPeer is a citation-grounded multi-agent peer-review assistant over a local scientific
corpus. Architecture follows the "blueprint for an AI co-reviewer" (see `docs/ARCHITECTURE.md`):

```
PDF -> GROBID + MinerU -> Qdrant (dense+BM25+ColPali) + LightRAG -> Agent SDK -> Review
```

## Hard invariants (never violate)

1. **Never fabricate a citation.** Every `[source:id]` in generated output MUST exist in the
   session's retrieval log. Enforced by `scholarpeer.eval.citation_grounding.verify()`.
2. **Never send corpus text to a cloud LLM without grounding.** All generation passes through
   `scholarpeer.synthesize.self_feedback` which pins retrieved passages.
3. **Never mutate indexed data in place.** Qdrant collections are immutable; re-index to update.
4. **Never swallow errors silently.** Use `structlog` with `log.error(event, exc_info=True)`.

## Retrieval API contracts

- **OpenAlex**: always include `mailto=<POLITE_EMAIL>` query param. Works endpoint:
  `https://api.openalex.org/works`. Rate: polite pool.
- **Semantic Scholar**: `https://api.semanticscholar.org/graph/v1`. Rate: ~100/5min without key.
  Use exponential backoff `10s -> 20s -> 40s` on 429.
- **Qdrant**: local `http://localhost:6333`. Default collection names: `sp_dense`, `sp_sparse`,
  `sp_colpali`. Never read from or write to a collection whose name does not start with `sp_`.

## Citation format (fixed)

- Local corpus: `[SP:<sha256-prefix-12>]` where prefix is from `PaperID` in `schemas.paper`.
- OpenAlex: `[OA:W<id>]`.
- Semantic Scholar: `[S2:<paperId>]`.
- DOI: `[DOI:<doi>]` only when the above are unavailable.

## Model routing (per blueprint)

| Role | Model | Why |
|------|-------|-----|
| Leader / planner | `claude-opus-4-7` | Deeper reasoning on review strategy |
| Specialist agents | `claude-sonnet-4-6` | Cheap + fast for section-level analysis |
| Formatting / refinement | `claude-haiku-4-5` | Cost-efficient final pass |

Use `model="inherit"` only for utility subagents with no routing preference.

## File conventions

- Package source: `src/scholarpeer/` (src-layout).
- Markdown corpus: configurable via `SP_CORPUS_DIR`, defaults to
  `F:/TREE_OF_THOUGHT/PHD/papers/literature_review`.
- PDF corpus: `SP_PDF_DIR`, defaults to `F:/TREE_OF_THOUGHT/PHD/papers/new_pdf`.
- Embeddings cache: `data/cache/embeddings/` (never committed).
- File naming: lowercase with hyphens (`self-feedback.py`, not `selfFeedback.py`).

## Test command

```bash
pytest -m "unit or integration"                    # default
pytest -m e2e                                      # requires docker-compose up
pytest --cov=src/scholarpeer --cov-report=term-missing
```

## Commit style

Conventional Commits only: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
Never include `Co-Authored-By` lines — global attribution is disabled in `~/.claude/settings.json`.

## When editing agent code

Every specialist agent in `src/scholarpeer/agents/specialists/` must:

1. Subclass `BaseSpecialist` from `agents/base.py`.
2. Declare its MARG role (`novelty | methodology | clarity | reproducibility | related_work`).
3. Return `ReviewerComment` objects (schema in `schemas/review.py`), never raw strings.
4. Pin its tool list via `allowed_tools=[...]` — no wildcard access.

## When editing retrieval code

All retrieval paths must emit a `RetrievalLog` entry so citation grounding can verify sources
later. Never call an embedding model or Qdrant directly — go through
`scholarpeer.retrieve.hybrid.HybridRetriever` so instrumentation is consistent.

## Further reading (read on demand only)

- `docs/ARCHITECTURE.md` — 5-layer diagram and dataflow.
- `docs/API_CONTRACTS.md` — OpenAlex / S2 / Unpaywall payload shapes.
- `docs/DEPLOYMENT.md` — Docker Compose setup for Qdrant + GROBID.
- `docs/EVAL.md` — ScholarQABench + MARG user-study replication.
