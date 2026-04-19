"""Unit tests for SectionChunker."""

from __future__ import annotations

import pytest

from scholarpeer.index.chunker import SectionChunker
from scholarpeer.schemas.paper import Paper, PaperSection


@pytest.mark.unit
def test_chunker_emits_abstract_first_if_present():
    paper = Paper(
        paper_id=Paper.make_id(title="foo"),
        title="foo",
        abstract="This is the abstract.",
        sections=(
            PaperSection(heading="Intro", level=1, text="Intro text", order=0),
        ),
    )
    chunks = SectionChunker().chunk_paper(paper)
    assert chunks[0].section == "abstract"
    assert chunks[1].section == "Intro"


@pytest.mark.unit
def test_chunker_windows_long_sections():
    long_text = "word " * 5000  # ~5000 tokens
    paper = Paper(
        paper_id=Paper.make_id(title="long"),
        title="long",
        sections=(PaperSection(heading="Methods", level=1, text=long_text, order=0),),
    )
    chunks = SectionChunker(chunk_tokens=512, overlap=64).chunk_paper(paper)
    assert len(chunks) > 1
    assert all(c.section == "Methods" for c in chunks)


@pytest.mark.unit
def test_chunker_empty_paper_produces_no_chunks():
    paper = Paper(paper_id=Paper.make_id(title="empty"), title="empty")
    assert SectionChunker().chunk_paper(paper) == []


@pytest.mark.unit
def test_chunker_assigns_monotonic_order():
    paper = Paper(
        paper_id=Paper.make_id(title="ord"),
        title="ord",
        abstract="abstract text",
        sections=(
            PaperSection(heading="Intro", level=1, text="intro", order=0),
            PaperSection(heading="Methods", level=1, text="methods", order=1),
        ),
    )
    chunks = SectionChunker().chunk_paper(paper)
    orders = [c.order_in_paper for c in chunks]
    assert orders == sorted(orders)
