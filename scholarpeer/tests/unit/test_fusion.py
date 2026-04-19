"""Unit tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from scholarpeer.retrieve.fusion import reciprocal_rank_fusion
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import Chunk, RetrievalHit


def _hit(chunk_id: str, rank: int, retriever: str = "dense") -> RetrievalHit:
    pid = Paper.make_id(title=chunk_id)
    return RetrievalHit(
        chunk=Chunk(
            chunk_id=chunk_id.ljust(12, "x")[:12],
            paper_id=pid,
            text=chunk_id,
            token_count=1,
            order_in_paper=0,
        ),
        score=1.0 / (rank + 1),
        rank=rank,
        retriever=retriever,
    )


@pytest.mark.unit
def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([], k=60, top_k=10) == []


@pytest.mark.unit
def test_rrf_respects_top_k():
    dense = [_hit(f"chunk_{i:02d}", rank=i, retriever="dense") for i in range(10)]
    merged = reciprocal_rank_fusion([dense], k=60, top_k=5)
    assert len(merged) == 5


@pytest.mark.unit
def test_rrf_boosts_items_ranked_highly_by_multiple_retrievers():
    # Chunk A is rank 0 in both lists -> should win.
    a_dense = _hit("A", rank=0, retriever="dense")
    a_sparse = _hit("A", rank=0, retriever="sparse")
    b_dense = _hit("B", rank=1, retriever="dense")
    c_sparse = _hit("C", rank=1, retriever="sparse")
    merged = reciprocal_rank_fusion(
        [[a_dense, b_dense], [a_sparse, c_sparse]],
        k=60,
        top_k=3,
    )
    ids = [h.chunk.chunk_id for h in merged]
    assert ids[0].startswith("A")


@pytest.mark.unit
def test_rrf_relabels_retriever_as_fused():
    dense = [_hit("X", rank=0, retriever="dense")]
    merged = reciprocal_rank_fusion([dense], k=60, top_k=1)
    assert merged[0].retriever == "fused"
