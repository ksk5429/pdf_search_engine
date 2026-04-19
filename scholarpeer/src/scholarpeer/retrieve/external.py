"""External citation sources: OpenAlex (corpus) + Semantic Scholar (citation graph).

OpenAlex backbone: 243M works, broadest coverage of any single source per the 2025
Monash study. Semantic Scholar provides citation edges and TLDRs as a complement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ExternalPaper:
    source: str  # "openalex" | "s2"
    id: str
    title: str
    abstract: str | None
    year: int | None
    authors: tuple[str, ...]
    doi: str | None
    cited_by: int
    url: str | None

    def citation_key(self) -> str:
        prefix = {"openalex": "OA", "s2": "S2"}[self.source]
        return f"{prefix}:{self.id}"


class OpenAlexClient:
    """https://api.openalex.org/works."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.openalex_base.rstrip("/")
        self._email = settings.polite_email

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        min_year: int = 2015,
        oa_only: bool = True,
    ) -> list[ExternalPaper]:
        params = {
            "search": query,
            "per_page": min(limit, 50),
            "select": "id,doi,title,abstract_inverted_index,publication_year,authorships,"
            "cited_by_count,primary_location",
            "mailto": self._email or "scholarpeer@example.com",
            "sort": "cited_by_count:desc",
        }
        filters = [f"from_publication_date:{min_year}-01-01"]
        if oa_only:
            filters.append("has_oa_accepted_or_published_version:true")
        params["filter"] = ",".join(filters)

        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{self._base}/works", params=params)
        r.raise_for_status()
        results = []
        for work in r.json().get("results", []):
            doi = work.get("doi") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            oa_id = (work.get("id") or "").rsplit("/", 1)[-1]
            abstract = _openalex_abstract(work.get("abstract_inverted_index"))
            authors = tuple(
                a.get("author", {}).get("display_name", "")
                for a in work.get("authorships", [])[:5]
            )
            url = (work.get("primary_location") or {}).get("landing_page_url")
            results.append(
                ExternalPaper(
                    source="openalex",
                    id=oa_id,
                    title=work.get("title") or "",
                    abstract=abstract,
                    year=work.get("publication_year"),
                    authors=authors,
                    doi=doi or None,
                    cited_by=int(work.get("cited_by_count") or 0),
                    url=url,
                )
            )
        return results


class SemanticScholarClient:
    """https://api.semanticscholar.org/graph/v1."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.s2_base.rstrip("/")

    def search(self, query: str, *, limit: int = 25, min_year: int = 2015) -> list[ExternalPaper]:
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,externalIds,openAccessPdf,citationCount",
            "year": f"{min_year}-",
        }
        for attempt in range(3):
            with httpx.Client(timeout=30.0) as client:
                r = client.get(f"{self._base}/paper/search", params=params)
            if r.status_code == 429:
                wait = 10 * (2**attempt)
                log.info("s2.rate_limited", wait_s=wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            return []

        out = []
        for paper in r.json().get("data", []):
            ext = paper.get("externalIds") or {}
            out.append(
                ExternalPaper(
                    source="s2",
                    id=paper.get("paperId") or "",
                    title=paper.get("title") or "",
                    abstract=paper.get("abstract"),
                    year=paper.get("year"),
                    authors=tuple(a.get("name", "") for a in paper.get("authors", [])[:5]),
                    doi=ext.get("DOI"),
                    cited_by=int(paper.get("citationCount") or 0),
                    url=(paper.get("openAccessPdf") or {}).get("url"),
                )
            )
        return out

    def citations_of(self, s2_paper_id: str, limit: int = 50) -> list[ExternalPaper]:
        """Return papers that cite ``s2_paper_id``."""
        with httpx.Client(timeout=30.0) as client:
            r = client.get(
                f"{self._base}/paper/{s2_paper_id}/citations",
                params={"limit": limit, "fields": "title,year,authors,externalIds,citationCount"},
            )
        if r.status_code != 200:
            return []
        out = []
        for entry in r.json().get("data", []):
            paper = entry.get("citingPaper") or {}
            out.append(
                ExternalPaper(
                    source="s2",
                    id=paper.get("paperId") or "",
                    title=paper.get("title") or "",
                    abstract=None,
                    year=paper.get("year"),
                    authors=tuple(a.get("name", "") for a in paper.get("authors", [])[:5]),
                    doi=(paper.get("externalIds") or {}).get("DOI"),
                    cited_by=int(paper.get("citationCount") or 0),
                    url=None,
                )
            )
        return out


def _openalex_abstract(inverted: dict | None) -> str | None:
    """OpenAlex returns abstracts as inverted index. Reconstruct to plain text."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for token, idxs in inverted.items():
        for i in idxs:
            positions[i] = token
    return " ".join(positions[i] for i in sorted(positions))
