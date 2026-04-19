"""Paper / Author / Section models. Immutable — update by creating a new instance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from scholarpeer.utils.hashing import paper_id_from_text

PaperID = Annotated[str, StringConstraints(min_length=12, max_length=12, pattern=r"^[a-f0-9]{12}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Author(_Frozen):
    name: str
    family: str | None = None
    given: str | None = None
    orcid: str | None = None
    affiliation: str | None = None


class PaperSection(_Frozen):
    """One logical section of a paper (abstract, intro, methods, etc.)."""

    heading: str
    level: int = Field(ge=1, le=6)
    text: str
    order: int = Field(ge=0)


class Paper(_Frozen):
    """Canonical paper representation produced by the ingest layer.

    ``paper_id`` is a stable 12-char sha256 prefix derived from DOI or normalized title.
    It is the only identifier downstream layers may assume is present.
    """

    paper_id: PaperID
    title: str
    abstract: str | None = None
    authors: tuple[Author, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    s2_paper_id: str | None = None
    arxiv_id: str | None = None

    # Content
    sections: tuple[PaperSection, ...] = ()
    references: tuple[str, ...] = ()

    # Provenance
    source_pdf: Path | None = None
    source_md: Path | None = None
    parser: str = "mineru+grobid"
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @classmethod
    def make_id(cls, title: str, doi: str | None = None) -> PaperID:
        return paper_id_from_text(title=title, doi=doi)

    def citation_key(self) -> str:
        """Return the canonical citation key for this paper in generated text."""
        return f"SP:{self.paper_id}"
