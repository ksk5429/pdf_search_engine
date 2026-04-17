"""Tests for pdf_search_engine core functions.

Run: python tests/test_search_engine.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set dummy email so module loads without error
os.environ.setdefault("POLITE_EMAIL", "test@example.com")

from pdf_search_engine import (
    title_hash,
    sanitize_filename,
    is_relevant,
    is_known,
    mark_known,
    DOMAIN_KEYWORDS,
)
from convert_to_markdown import (
    pdf_to_md_name,
    validate_markdown,
)


def test_title_hash_deterministic():
    """Same title should always produce same hash."""
    h1 = title_hash("Scour effects on offshore wind turbine foundations")
    h2 = title_hash("Scour effects on offshore wind turbine foundations")
    assert h1 == h2


def test_title_hash_case_insensitive():
    """Title hash should be case-insensitive."""
    h1 = title_hash("Scour Effects on Offshore Wind")
    h2 = title_hash("scour effects on offshore wind")
    assert h1 == h2


def test_title_hash_strips_punctuation():
    """Title hash should ignore punctuation."""
    h1 = title_hash("A Novel Approach: Scour Detection")
    h2 = title_hash("A Novel Approach Scour Detection")
    assert h1 == h2


def test_title_hash_length():
    """Title hash should be 16 characters."""
    h = title_hash("Any title here")
    assert len(h) == 16


def test_sanitize_filename_basic():
    """Should produce (Year Author) Title.pdf format."""
    result = sanitize_filename(
        "Lateral capacity of suction buckets",
        "10.1234/test",
        2024,
        ["Kyeong Sun Kim", "John Doe"],
    )
    assert result.startswith("(2024 Kim)")
    assert result.endswith(".pdf")
    assert "Lateral capacity" in result


def test_sanitize_filename_no_year():
    """Should use XXXX for unknown year."""
    result = sanitize_filename("Test title", "", None, ["Author"])
    assert "(XXXX Author)" in result


def test_sanitize_filename_no_authors():
    """Should use Unknown for missing authors."""
    result = sanitize_filename("Test title", "", 2024, [])
    assert "(2024 Unknown)" in result


def test_sanitize_filename_strips_illegal_chars():
    """Should remove characters illegal in filenames."""
    result = sanitize_filename('Title with "quotes" and <brackets>', "", 2024, ["Auth"])
    assert '"' not in result
    assert '<' not in result
    assert '>' not in result


def test_is_relevant_positive():
    """Should match domain keywords in title."""
    assert is_relevant("Scour effects on offshore wind turbine foundation")
    assert is_relevant("Finite element analysis of pile capacity")
    assert is_relevant("Machine learning for geotechnical prediction")
    assert is_relevant("Centrifuge modelling of suction caisson")


def test_is_relevant_negative():
    """Should reject non-domain titles."""
    assert not is_relevant("Quantum computing algorithms for optimization")
    assert not is_relevant("Social media marketing strategies")
    assert not is_relevant("")
    assert not is_relevant(None)


def test_domain_keywords_count():
    """Should have a substantial keyword set."""
    assert len(DOMAIN_KEYWORDS) >= 50


def test_is_known_and_mark_known():
    """Known paper tracking should work bidirectionally."""
    known = set()
    assert not is_known("10.1234/test", "Test Paper", known)
    mark_known("10.1234/test", "Test Paper", known)
    assert is_known("10.1234/test", "Test Paper", known)
    assert is_known("10.1234/test", "Different Title", known)  # DOI match
    assert is_known("", "Test Paper", known)  # title hash match


def test_pdf_to_md_name():
    """Should convert .pdf extension to .md."""
    assert pdf_to_md_name("(2024 Kim) Paper Title.pdf") == "(2024 Kim) Paper Title.md"


def test_validate_markdown_valid():
    """Valid markdown should pass validation."""
    para = "This is a proper paragraph with enough words and content to pass the quality validation checks in the converter. " * 3
    text = (para + "\n\n") * 5  # 5 paragraphs of substantial text
    is_valid, issues = validate_markdown(text, "test.pdf")
    assert is_valid, f"Should be valid but got issues: {issues}"
    assert len(issues) == 0


def test_validate_markdown_too_short():
    """Very short text should fail validation."""
    is_valid, issues = validate_markdown("Short.", "test.pdf")
    assert not is_valid
    assert any("too short" in i for i in issues)


def test_validate_markdown_too_few_words():
    """Text with too few words should fail validation."""
    is_valid, issues = validate_markdown("A " * 50, "test.pdf")  # 50 single-char words
    assert not is_valid


def test_validate_markdown_few_paragraphs():
    """Text with too few paragraphs should fail."""
    text = "This is one long paragraph with many words but no paragraph breaks at all " * 20
    is_valid, issues = validate_markdown(text, "test.pdf")
    assert not is_valid
    assert any("paragraph" in i for i in issues)


# ── Runner ────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_title_hash_deterministic,
        test_title_hash_case_insensitive,
        test_title_hash_strips_punctuation,
        test_title_hash_length,
        test_sanitize_filename_basic,
        test_sanitize_filename_no_year,
        test_sanitize_filename_no_authors,
        test_sanitize_filename_strips_illegal_chars,
        test_is_relevant_positive,
        test_is_relevant_negative,
        test_domain_keywords_count,
        test_is_known_and_mark_known,
        test_pdf_to_md_name,
        test_validate_markdown_valid,
        test_validate_markdown_too_short,
        test_validate_markdown_too_few_words,
        test_validate_markdown_few_paragraphs,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    print("Running pdf_search_engine tests...\n")
    success = run_all()
    sys.exit(0 if success else 1)
