"""Retrieval contracts: chunks, queries, hits, and the audit log used for citation grounding."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from scholarpeer.schemas.paper import PaperID


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CitationSource(str, Enum):
    LOCAL = "SP"
    OPENALEX = "OA"
    SEMANTIC_SCHOLAR = "S2"
    DOI = "DOI"


class Chunk(_Frozen):
    """A single retrievable unit — chunked, embedded, stored in Qdrant."""

    chunk_id: str = Field(min_length=12)
    paper_id: PaperID
    text: str
    section: str | None = None
    token_count: int = Field(ge=1)
    order_in_paper: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None


class RetrievalQuery(_Frozen):
    query: str
    top_k: int = Field(default=10, ge=1, le=200)
    filter_year_min: int | None = None
    filter_paper_ids: tuple[PaperID, ...] = ()
    mode: Literal["hybrid", "dense", "sparse", "visual", "external"] = "hybrid"


class RetrievalHit(_Frozen):
    """One retrieved chunk with ranking metadata."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=0)
    retriever: Literal["dense", "sparse", "colpali", "openalex", "s2", "fused"]
    rerank_score: float | None = None

    def citation_key(self, source: CitationSource = CitationSource.LOCAL) -> str:
        return f"{source.value}:{self.chunk.paper_id}"


class RetrievalLog(BaseModel):
    """Session-scoped audit log of every retrieval call.

    Used by citation-grounding to verify no ``[source:id]`` in generated output is fabricated.
    Mutable (appended during a session) but snapshotted before verification.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    hits: list[RetrievalHit] = Field(default_factory=list)
    queries: list[RetrievalQuery] = Field(default_factory=list)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def append(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> None:
        """Thread-safe append. Specialists run concurrently and share the log."""
        with self._lock:
            self.queries.append(query)
            self.hits.extend(hits)

    def cited_ids(self) -> set[str]:
        """Every ``[source:id]`` that would be valid to cite from this session."""
        ids: set[str] = set()
        for hit in self.hits:
            for src in CitationSource:
                ids.add(f"{src.value}:{hit.chunk.paper_id}")
        return ids
