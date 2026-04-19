"""Orchestrate GROBID (metadata) + MinerU (full-text) and emit ``Paper`` objects.

Output is written as paired ``<paper_id>.md`` + ``<paper_id>.json`` to
``data/corpus/`` so downstream layers (indexer, graph) can consume either form.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scholarpeer.config import get_settings
from scholarpeer.ingest.grobid import GrobidClient, GrobidMetadata
from scholarpeer.ingest.markdown_parser import MarkdownIngester
from scholarpeer.ingest.mineru import MinerUParser, MinerUResult
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import Paper

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    paper: Paper
    markdown_path: Path
    json_path: Path
    backend: str


class IngestPipeline:
    """PDF -> (GROBID metadata + MinerU full-text) -> Paper."""

    def __init__(
        self,
        out_dir: Path | None = None,
        grobid: GrobidClient | None = None,
        mineru: MinerUParser | None = None,
        use_grobid: bool = True,
    ) -> None:
        settings = get_settings()
        self._out = out_dir or settings.parsed_corpus_dir
        self._out.mkdir(parents=True, exist_ok=True)
        self._grobid = grobid or GrobidClient()
        self._mineru = mineru or MinerUParser()
        self._use_grobid = use_grobid and self._grobid.is_alive()
        if use_grobid and not self._use_grobid:
            log.warning("grobid.unavailable", url=self._grobid._base)
        self._md_ingester = MarkdownIngester()

    def ingest_pdf(self, pdf_path: Path) -> IngestResult:
        """Full dual-backend ingest of a single PDF."""
        log.info("ingest.start", pdf=str(pdf_path))

        metadata: GrobidMetadata | None = None
        if self._use_grobid:
            try:
                metadata = self._grobid.parse(pdf_path)
            except Exception as exc:  # noqa: BLE001 — keep pipeline resilient
                log.warning("grobid.failed", pdf=str(pdf_path), error=str(exc))

        mineru_result: MinerUResult = self._mineru.parse(pdf_path)

        # Merge: GROBID for metadata, MinerU for sections/full-text.
        title = (metadata.title if metadata else "") or _title_from_filename(pdf_path)
        paper = Paper(
            paper_id=Paper.make_id(title=title, doi=metadata.doi if metadata else None),
            title=title,
            abstract=(metadata.abstract if metadata else None),
            authors=(metadata.authors if metadata else ()),
            year=(metadata.year if metadata else None),
            venue=(metadata.venue if metadata else None),
            doi=(metadata.doi if metadata else None),
            sections=mineru_result.sections,
            references=(metadata.references if metadata else ()),
            source_pdf=pdf_path,
            parser=f"mineru={mineru_result.backend}+grobid={bool(metadata)}",
        )

        md_path, json_path = self._persist(paper, mineru_result.markdown)
        return IngestResult(
            paper=paper,
            markdown_path=md_path,
            json_path=json_path,
            backend=mineru_result.backend,
        )

    def ingest_markdown(self, md_path: Path) -> IngestResult | None:
        """Ingest an already-converted Markdown file from the existing corpus."""
        paper = self._md_ingester.ingest(md_path)
        if paper is None:
            return None
        md_dest, json_path = self._persist(paper, md_path.read_text(encoding="utf-8", errors="replace"))
        return IngestResult(paper=paper, markdown_path=md_dest, json_path=json_path, backend="markdown-only")

    def ingest_many(
        self,
        paths: Iterable[Path],
        *,
        workers: int = 4,
    ) -> list[IngestResult]:
        """Ingest a batch. PDFs run with thread parallelism (GROBID is IO-bound)."""
        paths = list(paths)
        log.info("ingest.batch.start", count=len(paths), workers=workers)
        results: list[IngestResult] = []

        # MinerU is GPU-bound — keep it sequential. GROBID is IO-bound — parallelize later.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for p in paths:
                if p.suffix.lower() == ".pdf":
                    futures.append(pool.submit(self.ingest_pdf, p))
                elif p.suffix.lower() == ".md":
                    futures.append(pool.submit(self.ingest_markdown, p))
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r:
                        results.append(r)
                except Exception as exc:  # noqa: BLE001
                    log.error("ingest.item.failed", error=str(exc), exc_info=True)

        log.info("ingest.batch.done", processed=len(results))
        return results

    # ── persistence ──────────────────────────────────────────────────────

    def _persist(self, paper: Paper, markdown: str) -> tuple[Path, Path]:
        # Defense in depth: although PaperID type-asserts hex-only, confirm the
        # resolved write path stays within self._out before touching the filesystem.
        out_root = self._out.resolve()
        base = (out_root / paper.paper_id).resolve()
        try:
            base.relative_to(out_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to write outside corpus dir: {base}") from exc
        md_path = base.with_suffix(".md")
        json_path = base.with_suffix(".json")
        md_path.write_text(markdown or "", encoding="utf-8")
        json_path.write_text(
            paper.model_dump_json(indent=2, exclude={"sections"}),
            encoding="utf-8",
        )
        # Store sections separately to keep paper JSON small
        sections_path = base.with_suffix(".sections.json")
        sections_path.write_text(
            json.dumps(
                [s.model_dump() for s in paper.sections],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return md_path, json_path


def _title_from_filename(path: Path) -> str:
    import re

    m = re.match(r"\(\d{4}\s+[^)]+\)\s*(.+)", path.stem)
    return (m.group(1) if m else path.stem).strip()
