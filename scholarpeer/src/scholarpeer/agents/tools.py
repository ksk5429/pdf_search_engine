"""Tool wrappers exposed to agents via the Claude Agent SDK.

All retrieval tools route through ``HybridRetriever`` so every hit lands in the
session ``RetrievalLog``. Agents may not import Qdrant or httpx directly.
"""

from __future__ import annotations

from typing import Any

from scholarpeer.logging import get_logger
from scholarpeer.retrieve.external import OpenAlexClient, SemanticScholarClient
from scholarpeer.retrieve.hybrid import HybridRetriever
from scholarpeer.schemas.retrieval import RetrievalLog, RetrievalQuery

log = get_logger(__name__)


def build_retrieval_tools(
    retriever: HybridRetriever,
    session_log: RetrievalLog,
    *,
    enable_external: bool = True,
) -> list[dict[str, Any]]:
    """Build SDK tool definitions. Returns a list of ``{name, description, handler}``.

    The Claude Agent SDK translates these into MCP-style tool schemas.
    """
    openalex = OpenAlexClient() if enable_external else None
    s2 = SemanticScholarClient() if enable_external else None

    def search_corpus(query: str, top_k: int = 10) -> dict[str, Any]:
        """Search the local scientific corpus via hybrid retrieval."""
        rq = RetrievalQuery(query=query, top_k=top_k)
        hits = retriever.search(rq, log_to=session_log, rerank=True)
        return {
            "hits": [
                {
                    "citation_key": f"SP:{h.chunk.paper_id}",
                    "paper_id": h.chunk.paper_id,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                    "score": h.rerank_score or h.score,
                }
                for h in hits
            ]
        }

    def search_openalex(query: str, limit: int = 10) -> dict[str, Any]:
        """Search OpenAlex for prior art outside the local corpus."""
        if openalex is None:
            return {"error": "external search disabled"}
        results = openalex.search(query, limit=limit)
        return {
            "results": [
                {
                    "citation_key": p.citation_key(),
                    "title": p.title,
                    "year": p.year,
                    "cited_by": p.cited_by,
                    "abstract": p.abstract,
                    "doi": p.doi,
                }
                for p in results
            ]
        }

    def search_s2(query: str, limit: int = 10) -> dict[str, Any]:
        """Search Semantic Scholar (citation graph + TLDRs)."""
        if s2 is None:
            return {"error": "external search disabled"}
        results = s2.search(query, limit=limit)
        return {
            "results": [
                {
                    "citation_key": p.citation_key(),
                    "title": p.title,
                    "year": p.year,
                    "cited_by": p.cited_by,
                    "doi": p.doi,
                    "abstract": p.abstract,
                }
                for p in results
            ]
        }

    return [
        {
            "name": "search_corpus",
            "description": (
                "Hybrid semantic + keyword search over the local scientific corpus. "
                "Returns chunks with citation keys of the form SP:<paper_id>. Use this "
                "first; only fall back to OpenAlex/S2 if local coverage is insufficient."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
            "handler": search_corpus,
        },
        {
            "name": "search_openalex",
            "description": (
                "Search OpenAlex (243M works) for prior art. Returns OA:<id> citation "
                "keys. Useful for novelty and related-work checks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
            "handler": search_openalex,
        },
        {
            "name": "search_s2",
            "description": "Search Semantic Scholar for citation-graph context.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
            "handler": search_s2,
        },
    ]
