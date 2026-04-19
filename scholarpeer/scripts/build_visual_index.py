"""Build the ColPali visual index over every PDF in ``SP_PDF_DIR``.

Resilient for long overnight runs:
  - Skips papers already indexed (checks Qdrant payload count).
  - Caps per-PDF pages (default 40) to keep runtime bounded.
  - Writes one log line per PDF to ``visual_index.log`` + structured JSONL to
    ``visual_index_results.jsonl`` so you can resume / audit after a crash.
  - Per-PDF errors are logged but don't abort the run.

Usage:
  python scripts/build_visual_index.py                      # all PDFs in SP_PDF_DIR
  python scripts/build_visual_index.py --limit 20           # smoke test
  python scripts/build_visual_index.py --pdf-dir path/      # override
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.index.colpali_indexer import ColPaliIndexer, VisualIndexResult
from scholarpeer.index.indexer import load_papers_from_corpus
from scholarpeer.logging import get_logger

log = get_logger(__name__)


def _paper_id_lookup(corpus_dir: Path) -> dict[str, str]:
    """Map source-PDF filename stem -> paper_id so we can match PDFs to ingested papers."""
    papers = load_papers_from_corpus(corpus_dir)
    mapping: dict[str, str] = {}
    for p in papers:
        if p.source_pdf:
            mapping[p.source_pdf.stem] = p.paper_id
        if p.source_md:
            mapping[p.source_md.stem] = p.paper_id
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="ScholarPeer: build ColPali visual index")
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None, help="Parsed corpus dir (for paper_id lookup)")
    parser.add_argument("--limit", type=int, default=0, help="Index at most N PDFs (0 = all)")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--page-batch", type=int, default=4)
    parser.add_argument("--out-log", type=Path, default=Path("visual_index.log"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("visual_index_results.jsonl"))
    args = parser.parse_args()

    settings = get_settings()
    pdf_dir = (args.pdf_dir or settings.pdf_dir).resolve()
    corpus = (args.corpus or settings.parsed_corpus_dir).resolve()

    log.info("visual.start", pdf_dir=str(pdf_dir), corpus=str(corpus))
    lookup = _paper_id_lookup(corpus)
    log.info("visual.corpus_loaded", papers=len(lookup))

    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    log.info("visual.pdfs_found", n=len(pdfs))

    indexer = ColPaliIndexer(max_pages_per_pdf=args.max_pages, page_batch=args.page_batch)

    out_jsonl = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    stats = {
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "no_paper_id": 0,
        "pages_total": 0,
    }

    with out_jsonl.open("a", encoding="utf-8") as out:
        for i, pdf in enumerate(pdfs, 1):
            paper_id = lookup.get(pdf.stem)
            if paper_id is None:
                # Fall back to a stable id derived from the filename so the PDF is still indexed.
                from scholarpeer.schemas.paper import Paper

                paper_id = Paper.make_id(title=pdf.stem)
                stats["no_paper_id"] += 1

            try:
                result = indexer.index_pdf(pdf, paper_id)
            except Exception as exc:  # noqa: BLE001 — per-paper isolation
                log.error("visual.pdf_failed", pdf=pdf.name, error=str(exc), exc_info=True)
                stats["failed"] += 1
                continue

            if result.skipped:
                stats["skipped"] += 1
            elif result.error:
                stats["failed"] += 1
            else:
                stats["indexed"] += 1
                stats["pages_total"] += result.pages_indexed

            out.write(json.dumps(_result_to_dict(result), ensure_ascii=False) + "\n")
            out.flush()

            if i % 10 == 0 or i == len(pdfs):
                elapsed = time.time() - start_all
                rate = i / elapsed if elapsed else 0
                eta_s = (len(pdfs) - i) / rate if rate else -1
                log.info(
                    "visual.progress",
                    done=i,
                    total=len(pdfs),
                    rate_per_s=round(rate, 2),
                    eta_min=round(eta_s / 60, 1) if eta_s > 0 else None,
                    **stats,
                )

    elapsed = time.time() - start_all
    log.info("visual.done", total_s=round(elapsed, 1), **stats)
    return 0 if stats["failed"] == 0 else 2


def _result_to_dict(r: VisualIndexResult) -> dict:
    d = asdict(r)
    d["pdf_path"] = str(d["pdf_path"])
    return d


if __name__ == "__main__":
    raise SystemExit(main())
