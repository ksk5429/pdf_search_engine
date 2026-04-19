"""Layer 3 — Retrieve. Hybrid dense+sparse + rerank, plus external citation sources."""

from scholarpeer.retrieve.external import OpenAlexClient, SemanticScholarClient
from scholarpeer.retrieve.fusion import reciprocal_rank_fusion
from scholarpeer.retrieve.hybrid import HybridRetriever
from scholarpeer.retrieve.rerank import CrossEncoderReranker

__all__ = [
    "CrossEncoderReranker",
    "HybridRetriever",
    "OpenAlexClient",
    "SemanticScholarClient",
    "reciprocal_rank_fusion",
]
