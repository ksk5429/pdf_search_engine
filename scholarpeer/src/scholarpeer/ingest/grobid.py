"""GROBID HTTP client — used for fast, reliable metadata + reference extraction.

GROBID still beats VLM-based parsers on bibliography structure per independent 2025
benchmarks. We use it for metadata/references and delegate full-text to MinerU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import Author

log = get_logger(__name__)

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass(frozen=True)
class GrobidMetadata:
    title: str
    abstract: str | None
    authors: tuple[Author, ...]
    year: int | None
    doi: str | None
    venue: str | None
    references: tuple[str, ...]
    raw_tei: str


class GrobidClient:
    """Thin async client for a local GROBID service."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self._base = (base_url or settings.grobid_url).rstrip("/")
        self._timeout = timeout

    def is_alive(self) -> bool:
        try:
            r = httpx.get(f"{self._base}/api/isalive", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def process_fulltext(self, pdf_path: Path) -> str:
        """POST PDF to ``/api/processFulltextDocument``. Returns TEI-XML string."""
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)

        url = f"{self._base}/api/processFulltextDocument"
        params = {
            "consolidateCitations": "1",
            "consolidateHeader": "1",
            "includeRawCitations": "1",
            "includeRawAffiliations": "1",
        }
        log.debug("grobid.request", pdf=str(pdf_path), url=url)
        with pdf_path.open("rb") as fh, httpx.Client(timeout=self._timeout) as client:
            files = {"input": (pdf_path.name, fh, "application/pdf")}
            resp = client.post(url, files=files, data=params)
        resp.raise_for_status()
        return resp.text

    def parse(self, pdf_path: Path) -> GrobidMetadata:
        """Full pipeline: PDF -> TEI -> ``GrobidMetadata``."""
        tei_xml = self.process_fulltext(pdf_path)
        return self.parse_tei(tei_xml)

    @staticmethod
    def parse_tei(tei_xml: str) -> GrobidMetadata:
        root = etree.fromstring(tei_xml.encode("utf-8"))
        header = root.find(".//tei:teiHeader", _TEI_NS)

        def _xp(node: Any, expr: str) -> str | None:
            if node is None:
                return None
            found = node.xpath(expr, namespaces=_TEI_NS)
            if not found:
                return None
            elem = found[0]
            text = elem.text if hasattr(elem, "text") else str(elem)
            return text.strip() if text else None

        title = _xp(header, ".//tei:titleStmt/tei:title[@type='main']") or ""
        abstract_node = root.find(".//tei:profileDesc/tei:abstract", _TEI_NS)
        abstract = _clean_text(abstract_node) if abstract_node is not None else None

        authors: list[Author] = []
        for author_elem in root.findall(".//tei:sourceDesc//tei:author", _TEI_NS):
            persname = author_elem.find(".//tei:persName", _TEI_NS)
            if persname is None:
                continue
            given = _xp(persname, ".//tei:forename[@type='first']")
            family = _xp(persname, ".//tei:surname")
            if not family and not given:
                continue
            name = " ".join(p for p in (given, family) if p)
            authors.append(Author(name=name, given=given, family=family))

        year_str = _xp(root, ".//tei:publicationStmt//tei:date[@type='published']/@when")
        year = int(year_str[:4]) if year_str and year_str[:4].isdigit() else None

        doi = _xp(root, ".//tei:idno[@type='DOI']")
        venue = _xp(root, ".//tei:monogr//tei:title[@level='j']")

        references: list[str] = []
        for bibl in root.findall(".//tei:listBibl/tei:biblStruct", _TEI_NS):
            raw = _clean_text(bibl)
            if raw:
                references.append(raw)

        return GrobidMetadata(
            title=title,
            abstract=abstract,
            authors=tuple(authors),
            year=year,
            doi=doi,
            venue=venue,
            references=tuple(references),
            raw_tei=tei_xml,
        )


def _clean_text(node: etree._Element) -> str:
    """Collapse whitespace and strip TEI markup."""
    text = " ".join(node.itertext())
    return " ".join(text.split())
