"""Unit tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholarpeer.schemas.paper import Author, Paper
from scholarpeer.schemas.retrieval import Chunk, RetrievalHit, RetrievalLog, RetrievalQuery
from scholarpeer.schemas.review import Review, ReviewerComment, ReviewSeverity, SpecialistRole


@pytest.mark.unit
def test_paper_id_is_deterministic():
    a = Paper.make_id(title="A Study of X", doi="10.1/abc")
    b = Paper.make_id(title="A Study of X", doi="10.1/abc")
    c = Paper.make_id(title="different title", doi="10.1/abc")
    assert a == b
    assert a == c  # DOI takes precedence


@pytest.mark.unit
def test_paper_id_stable_across_whitespace_and_case():
    a = Paper.make_id(title="A Study of X")
    b = Paper.make_id(title="  a STUDY  of x  ")
    assert a == b


@pytest.mark.unit
def test_paper_is_frozen():
    pid = Paper.make_id(title="foo")
    p = Paper(paper_id=pid, title="foo")
    with pytest.raises(ValidationError):
        p.title = "bar"  # type: ignore[misc]


@pytest.mark.unit
def test_retrieval_log_tracks_cited_ids():
    pid = Paper.make_id(title="foo")
    chunk = Chunk(
        chunk_id="abc123def456",
        paper_id=pid,
        text="content",
        token_count=2,
        order_in_paper=0,
    )
    hit = RetrievalHit(chunk=chunk, score=0.9, rank=0, retriever="dense")
    q = RetrievalQuery(query="foo", top_k=1)
    log = RetrievalLog(session_id="s1")
    log.append(q, [hit])
    ids = log.cited_ids()
    assert f"SP:{pid}" in ids
    assert f"OA:{pid}" in ids


@pytest.mark.unit
def test_reviewer_comment_evidence_required_by_convention():
    c = ReviewerComment(
        role=SpecialistRole.NOVELTY,
        severity=ReviewSeverity.MAJOR,
        comment="Overlaps with Smith 2020.",
        evidence_citations=("SP:abcdef012345",),
    )
    assert c.cited_ids() == {"SP:abcdef012345"}


@pytest.mark.unit
def test_review_collects_all_cited_ids():
    comments = [
        ReviewerComment(
            role=SpecialistRole.NOVELTY,
            severity=ReviewSeverity.MAJOR,
            comment="First novelty concern with citation.",
            evidence_citations=("SP:aaaaaaaaaaaa",),
        ),
        ReviewerComment(
            role=SpecialistRole.METHODOLOGY,
            severity=ReviewSeverity.MINOR,
            comment="Second methodology concern with citation.",
            evidence_citations=("SP:bbbbbbbbbbbb",),
        ),
    ]
    r = Review(
        target_paper_id=Paper.make_id(title="t"),
        target_title="t",
        summary="s",
        comments=comments,
    )
    assert r.all_cited_ids() == {"SP:aaaaaaaaaaaa", "SP:bbbbbbbbbbbb"}


@pytest.mark.unit
def test_author_accepts_minimal():
    a = Author(name="Turing")
    assert a.name == "Turing"
