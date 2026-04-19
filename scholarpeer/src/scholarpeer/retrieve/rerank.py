"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default).

Reranking is optional but produces the biggest single-step quality gain per the
BEIR/MTEB literature — ~5-10 points on recall@10 for scientific queries.
"""

from __future__ import annotations

from collections.abc import Sequence

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger
from scholarpeer.schemas.retrieval import RetrievalHit

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.rerank_model
        self._device = device or settings.device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker

        log.info("rerank.load", model=self._model_name)
        self._model = FlagReranker(
            self._model_name,
            use_fp16=(self._device != "cpu"),
            device=self._device,
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        self._load()
        pairs = [[query, h.chunk.text] for h in hits]
        scores = self._model.compute_score(pairs, normalize=True)  # type: ignore[union-attr]
        if isinstance(scores, float):
            scores = [scores]
        reranked = [
            h.model_copy(update={"rerank_score": float(s), "retriever": "fused"})
            for h, s in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        k = top_k or len(reranked)
        return [h.model_copy(update={"rank": i}) for i, h in enumerate(reranked[:k])]
