"""Integration test: round-trip a paper through the indexer + hybrid retriever.

Requires a live local Qdrant (`docker compose up -d qdrant`) and downloads model
weights on first run. Skipped automatically when Qdrant is unreachable.
"""

from __future__ import annotations

import pytest

from scholarpeer.config import get_settings
from scholarpeer.schemas.paper import Paper, PaperSection


@pytest.fixture(scope="module")
def qdrant_available() -> bool:
    import httpx

    try:
        r = httpx.get(f"{get_settings().qdrant_url}/", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.integration
def test_index_and_search_roundtrip(qdrant_available):
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")

    from scholarpeer.index.indexer import CorpusIndexer
    from scholarpeer.retrieve.hybrid import HybridRetriever
    from scholarpeer.schemas.retrieval import RetrievalLog, RetrievalQuery

    paper = Paper(
        paper_id=Paper.make_id(title="roundtrip test", doi="10.test/rt"),
        title="roundtrip test",
        abstract="We investigate scour around monopile foundations in sandy seabeds.",
        sections=(
            PaperSection(
                heading="Method",
                level=1,
                text=(
                    "Local scour depth is measured using high-frequency sonar. "
                    "We compare the field data against DNV-RP-C212 predictions."
                ),
                order=0,
            ),
        ),
    )

    indexer = CorpusIndexer(collection="sp_dense_test")
    indexer._collection = "sp_dense_test"  # noqa: SLF001
    stats = indexer.index([paper])
    assert stats[paper.paper_id] > 0

    retriever = HybridRetriever(collection="sp_dense_test")
    log_ = RetrievalLog(session_id="it")
    hits = retriever.search(
        RetrievalQuery(query="scour monopile sonar", top_k=5),
        log_to=log_,
        rerank=False,
    )
    assert hits, "expected at least one hit"
    assert hits[0].chunk.paper_id == paper.paper_id
