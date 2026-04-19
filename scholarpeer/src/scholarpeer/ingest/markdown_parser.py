"""Reuse the existing ``literature_review/*.md`` corpus directly.

The user already has ~500 converted Markdown files. We ingest them without re-parsing,
falling back to heuristics for missing metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

from scholarpeer.ingest.mineru import _sections_from_markdown
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import Author, Paper

log = get_logger(__name__)

_FILENAME_PAT = re.compile(r"\((\d{4})\s+([^)]+)\)\s*(.+?)(?:\.md)?$")


class MarkdownIngester:
    """Convert an existing Markdown file to a ``Paper`` object via filename heuristics."""

    def ingest(self, md_path: Path) -> Paper | None:
        if not md_path.is_file():
            log.warning("md.missing", path=str(md_path))
            return None
        stem = md_path.stem
        match = _FILENAME_PAT.match(stem)
        if match:
            year_str, author_str, title = match.groups()
            year: int | None = int(year_str)
            authors = self._parse_authors(author_str)
        else:
            title = stem
            year = None
            authors = ()

        text = md_path.read_text(encoding="utf-8", errors="replace")
        sections = _sections_from_markdown(text)
        abstract = _extract_abstract(sections)

        paper_id = Paper.make_id(title=title)
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            sections=sections,
            source_md=md_path,
            parser="markdown-only",
        )

    @staticmethod
    def _parse_authors(raw: str) -> tuple[Author, ...]:
        parts = re.split(r"\s+(?:et al\.?|&|and)\s+|,", raw, flags=re.IGNORECASE)
        authors: list[Author] = []
        for name in parts:
            name = name.strip(" .")
            if name and len(name) > 1:
                authors.append(Author(name=name, family=name))
        return tuple(authors)


def _extract_abstract(sections: tuple, max_chars: int = 1500) -> str | None:
    for sec in sections:
        if "abstract" in sec.heading.lower():
            return sec.text[:max_chars].strip() or None
    # Fall back to first paragraph of first section
    if sections:
        return sections[0].text[:max_chars].strip() or None
    return None
