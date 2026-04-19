"""Unit tests for citation grounding."""

from __future__ import annotations

import pytest

from scholarpeer.eval.citation_grounding import verify_grounding
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import Chunk, RetrievalHit, RetrievalLog, RetrievalQuery
from scholarpeer.schemas.review import Review, ReviewerComment, ReviewSeverity, SpecialistRole


def _log_with(paper_id: str) -> RetrievalLog:
    chunk = Chunk(
        chunk_id="abc123def456",
        paper_id=paper_id,
        text="content",
        token_count=1,
        order_in_paper=0,
    )
    log = RetrievalLog(session_id="s")
    log.append(
        RetrievalQuery(query="q", top_k=1),
        [RetrievalHit(chunk=chunk, score=0.9, rank=0, retriever="dense")],
    )
    return log


@pytest.mark.unit
def test_grounded_review_passes():
    pid = Paper.make_id(title="t")
    log = _log_with(pid)
    comment = ReviewerComment(
        role=SpecialistRole.NOVELTY,
        severity=ReviewSeverity.MAJOR,
        comment="The claim overlaps with earlier work.",
        evidence_citations=(f"SP:{pid}",),
    )
    review = Review(target_paper_id=pid, target_title="t", summary="x", comments=[comment])
    report = verify_grounding(review, log)
    assert report.grounded
    assert report.grounding_rate == 1.0


@pytest.mark.unit
def test_ungrounded_citation_flagged():
    pid = Paper.make_id(title="t")
    log = _log_with(pid)
    comment = ReviewerComment(
        role=SpecialistRole.NOVELTY,
        severity=ReviewSeverity.MAJOR,
        comment="Inline [SP:fffffffffff1] should be flagged.",
        evidence_citations=(),
    )
    review = Review(target_paper_id=pid, target_title="t", summary="x", comments=[comment])
    report = verify_grounding(review, log)
    assert not report.grounded
    assert "SP:fffffffffff1" in report.invalid_citations


@pytest.mark.unit
def test_multiple_ungrounded_deduped():
    pid = Paper.make_id(title="t")
    log = _log_with(pid)
    comments = [
        ReviewerComment(
            role=SpecialistRole.NOVELTY,
            severity=ReviewSeverity.MINOR,
            comment="A minor concern with two duplicate citations.",
            evidence_citations=("SP:nope00000001", "SP:nope00000001"),
        )
    ]
    review = Review(target_paper_id=pid, target_title="t", summary="s", comments=comments)
    report = verify_grounding(review, log)
    assert report.invalid_citations == ("SP:nope00000001",)
    assert report.total == 1
