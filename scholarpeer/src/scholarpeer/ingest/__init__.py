"""Layer 1 — Ingest. Convert PDFs to ``Paper`` objects with full text + metadata."""

from scholarpeer.ingest.grobid import GrobidClient, GrobidMetadata
from scholarpeer.ingest.markdown_parser import MarkdownIngester
from scholarpeer.ingest.mineru import MinerUParser, MinerUResult
from scholarpeer.ingest.pipeline import IngestPipeline, IngestResult

__all__ = [
    "GrobidClient",
    "GrobidMetadata",
    "IngestPipeline",
    "IngestResult",
    "MarkdownIngester",
    "MinerUParser",
    "MinerUResult",
]
