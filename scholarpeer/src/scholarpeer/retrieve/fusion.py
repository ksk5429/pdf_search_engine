"""Score-free rank fusion. Reciprocal Rank Fusion (RRF) is the default.

RRF is robust when scores from different retrievers are on incomparable scales
(cosine similarity vs BM25 vs ColPali late-interaction).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from scholarpeer.schemas.retrieval import RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalHit]],
    *,
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievalHit]:
    """Standard RRF: score = sum_i 1 / (k + rank_i). Higher is better."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, RetrievalHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            key = hit.chunk.chunk_id
            scores[key] += 1.0 / (k + rank + 1)
            if key not in first_seen:
                first_seen[key] = hit

    merged: list[RetrievalHit] = []
    for i, (chunk_id, fused_score) in enumerate(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ):
        source = first_seen[chunk_id]
        merged.append(
            source.model_copy(update={"score": fused_score, "rank": i, "retriever": "fused"})
        )
    return merged
