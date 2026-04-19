"""Unit tests for the Markdown ingester — no IO, works on temp paths."""

from __future__ import annotations

import pytest

from scholarpeer.ingest.markdown_parser import MarkdownIngester


MD_DOC = """# Abstract

We study X using Y and report Z.

# Introduction

Prior work on X is well-established [1,2]. Our contribution is Q.

# Methods

We solve equation (1) with a finite-element scheme.
"""


@pytest.mark.unit
def test_ingest_parses_filename_metadata(tmp_path):
    p = tmp_path / "(2023 Smith) A Study of X.md"
    p.write_text(MD_DOC, encoding="utf-8")

    paper = MarkdownIngester().ingest(p)

    assert paper is not None
    assert paper.year == 2023
    assert paper.authors[0].name == "Smith"
    assert paper.title == "A Study of X"
    assert paper.abstract and "X" in paper.abstract
    headings = {s.heading for s in paper.sections}
    assert {"Abstract", "Introduction", "Methods"} <= headings


@pytest.mark.unit
def test_ingest_missing_file_returns_none(tmp_path):
    assert MarkdownIngester().ingest(tmp_path / "nope.md") is None
