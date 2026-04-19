"""Hybrid retriever: dense + sparse -> RRF fusion -> cross-encoder rerank.

Every call emits entries into the session's ``RetrievalLog`` so downstream citation
grounding can verify that the agent didn't hallucinate a source.
"""

from __future__ import annotations

from scholarpeer.config import get_settings
from scholarpeer.index.embeddings import DenseEmbedder, SparseEmbedder
from scholarpeer.index.qdrant_client import QdrantStore
from scholarpeer.logging import get_logger
from scholarpeer.retrieve.fusion import reciprocal_rank_fusion
from scholarpeer.retrieve.rerank import CrossEncoderReranker
from scholarpeer.schemas.retrieval import Chunk, RetrievalHit, RetrievalLog, RetrievalQuery

log = get_logger(__name__)


class HybridRetriever:
    """Reference retrieval path. All agent tool calls should go through this class."""

    def __init__(
        self,
        store: QdrantStore | None = None,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or QdrantStore()
        self._dense = dense or DenseEmbedder()
        self._sparse = sparse or SparseEmbedder()
        self._reranker = reranker or CrossEncoderReranker()
        self._collection = collection or settings.collection_dense
        self._rrf_k = settings.rrf_k

    def search(
        self,
        query: RetrievalQuery | str,
        *,
        log_to: RetrievalLog | None = None,
        rerank: bool = True,
    ) -> list[RetrievalHit]:
        if isinstance(query, str):
            query = RetrievalQuery(query=query)

        settings = get_settings()
        dense_vec = self._dense.encode([query.query], is_query=True)[0]
        sparse_vec = self._sparse.encode([query.query])[0]

        dense_points = self._store.search_dense(
            self._collection,
            dense_vec,
            top_k=settings.top_k_dense,
            filter_paper_ids=query.filter_paper_ids,
        )
        sparse_points = self._store.search_sparse(
            self._collection,
            sparse_vec,
            top_k=settings.top_k_sparse,
            filter_paper_ids=query.filter_paper_ids,
        )

        dense_hits = [_point_to_hit(p, rank=i, retriever="dense") for i, p in enumerate(dense_points)]
        sparse_hits = [_point_to_hit(p, rank=i, retriever="sparse") for i, p in enumerate(sparse_points)]

        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=self._rrf_k,
            top_k=max(settings.top_k_rerank * 3, query.top_k),
        )

        if rerank and fused:
            fused = self._reranker.rerank(query.query, fused, top_k=query.top_k)
        else:
            fused = fused[: query.top_k]

        if log_to is not None:
            log_to.append(query, fused)
        return fused


def _point_to_hit(point, *, rank: int, retriever: str) -> RetrievalHit:  # type: ignore[no-untyped-def]
    payload = point.payload or {}
    chunk = Chunk(
        chunk_id=payload["chunk_id"],
        paper_id=payload["paper_id"],
        text=payload["text"],
        section=payload.get("section"),
        token_count=int(payload.get("token_count", 1)),
        order_in_paper=int(payload.get("order_in_paper", 0)),
    )
    return RetrievalHit(chunk=chunk, score=float(point.score), rank=rank, retriever=retriever)
