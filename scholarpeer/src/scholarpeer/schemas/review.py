"""Review output contract — what the agent pipeline emits.

Structured to mirror the MARG user-study rubric (specificity, accuracy, helpfulness)
so evaluations are straightforward.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from scholarpeer.schemas.paper import PaperID


class SpecialistRole(str, Enum):
    NOVELTY = "novelty"
    METHODOLOGY = "methodology"
    CLARITY = "clarity"
    REPRODUCIBILITY = "reproducibility"
    RELATED_WORK = "related_work"


class ReviewSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"
    STRENGTH = "strength"


class ReviewerComment(BaseModel):
    """One comment from one specialist, with inline grounded citations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: SpecialistRole
    severity: ReviewSeverity
    section_ref: str | None = None
    comment: str = Field(min_length=10)
    evidence_citations: tuple[str, ...] = Field(
        default=(),
        description="Citation keys like 'SP:abc123' or 'OA:W...' that back this comment.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)

    def cited_ids(self) -> set[str]:
        return set(self.evidence_citations)


class Review(BaseModel):
    """Top-level review artifact. Returned by the pipeline and serialized to disk."""

    model_config = ConfigDict(extra="forbid")

    target_paper_id: PaperID
    target_title: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    comments: list[ReviewerComment] = Field(default_factory=list)

    recommendation: str = Field(
        default="",
        description="e.g., 'accept', 'minor revision', 'major revision', 'reject'.",
    )
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    session_id: str = ""
    model_routing: dict[str, str] = Field(default_factory=dict)

    def all_cited_ids(self) -> set[str]:
        return {cid for c in self.comments for cid in c.cited_ids()}
