"""One-shot ingest script for both PDF and Markdown corpora.

Runs the existing MD corpus (``literature_review/``) first — those files are already
converted and need no GROBID/MinerU. Then runs any new PDFs in ``new_pdf/`` through
the dual-backend pipeline.
"""

from __future__ import annotations

from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.ingest.pipeline import IngestPipeline
from scholarpeer.logging import get_logger

log = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    pipeline = IngestPipeline()

    tasks: list[Path] = []
    if settings.corpus_dir.exists():
        tasks.extend(sorted(settings.corpus_dir.glob("*.md")))
        log.info("md.count", n=sum(1 for _ in settings.corpus_dir.glob("*.md")))
    if settings.pdf_dir.exists():
        tasks.extend(sorted(settings.pdf_dir.glob("*.pdf")))
        log.info("pdf.count", n=sum(1 for _ in settings.pdf_dir.glob("*.pdf")))

    log.info("ingest.all.start", total=len(tasks))
    results = pipeline.ingest_many(tasks, workers=4)
    log.info("ingest.all.done", processed=len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
