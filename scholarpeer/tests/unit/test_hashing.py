"""Unit tests for hashing utilities."""

from __future__ import annotations

import pytest

from scholarpeer.utils.hashing import content_sha256, paper_id_from_text, short_hash


@pytest.mark.unit
def test_content_sha256_stable():
    assert content_sha256("hello") == content_sha256(b"hello")


@pytest.mark.unit
def test_short_hash_length():
    assert len(short_hash("x")) == 12
    assert len(short_hash("x", length=8)) == 8


@pytest.mark.unit
def test_paper_id_doi_overrides_title():
    assert paper_id_from_text("A", doi="10/x") == paper_id_from_text("B", doi="10/x")
    assert paper_id_from_text("A") != paper_id_from_text("B")
