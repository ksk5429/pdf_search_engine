"""Pydantic schemas — the contracts between layers."""

from scholarpeer.schemas.paper import Author, Paper, PaperID, PaperSection
from scholarpeer.schemas.retrieval import (
    Chunk,
    CitationSource,
    RetrievalHit,
    RetrievalLog,
    RetrievalQuery,
)
from scholarpeer.schemas.review import (
    Review,
    ReviewerComment,
    ReviewSeverity,
    SpecialistRole,
)

__all__ = [
    # paper
    "Author",
    "Paper",
    "PaperID",
    "PaperSection",
    # retrieval
    "Chunk",
    "CitationSource",
    "RetrievalHit",
    "RetrievalLog",
    "RetrievalQuery",
    # review
    "Review",
    "ReviewerComment",
    "ReviewSeverity",
    "SpecialistRole",
]
