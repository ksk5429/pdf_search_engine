"""
PDF-to-Markdown Conversion Pipeline
====================================
Converts academic PDFs to structured markdown using pymupdf4llm.
Handles batch processing, deduplication, and quality validation.

Usage:
  python convert_to_markdown.py                              # convert all new PDFs
  python convert_to_markdown.py --dry-run                    # preview only
  python convert_to_markdown.py --single "file.pdf"          # convert one file
  python convert_to_markdown.py --recheck                    # re-validate existing
  python convert_to_markdown.py --pdf-dir ./pdfs --output-dir ./markdown
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path

import pymupdf4llm

# ── Configuration ──────────────────────────────────────────────────────────
PDF_DIR = Path(os.environ.get("PDF_DOWNLOAD_DIR", Path(__file__).parent / "downloads"))
OUTPUT_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", Path(__file__).parent / "markdown"))
CONVERSION_LOG = PDF_DIR / "conversion_log.json"

# Quality thresholds
MIN_MARKDOWN_LENGTH = 500       # minimum chars for valid conversion
MIN_WORD_COUNT = 100            # minimum words
MAX_GARBLED_RATIO = 0.15        # max ratio of non-ASCII chars (excluding CJK)
MIN_PARAGRAPH_COUNT = 3         # minimum number of paragraphs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PDF_DIR / "conversion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Filename Mapping ───────────────────────────────────────────────────────

def pdf_to_md_name(pdf_name):
    """Convert PDF filename to literature_review markdown filename.

    Input:  (2020 Suryasentana) A Winkler model for suction caissons.pdf
    Output: (2020 Suryasentana) A Winkler model for suction caissons.md

    Also handles legacy naming convention:
    Input:  (2020 Suryasentana) Title.pdf
    Output: (Suryasentana 2020)Title.md  (if needed for matching)
    """
    return pdf_name.replace(".pdf", ".md")


def find_existing_md(pdf_name, output_dir):
    """Check if a markdown version already exists (exact or fuzzy match)."""
    md_name = pdf_to_md_name(pdf_name)
    exact = output_dir / md_name
    if exact.exists():
        return exact

    # Try fuzzy match: extract author and year, search for similar files
    match = re.match(r'\((\d{4})\s+(\w+)\)', pdf_name)
    if match:
        year, author = match.groups()
        # Check both naming conventions
        for f in output_dir.iterdir():
            if f.suffix != ".md":
                continue
            fname = f.name.lower()
            if author.lower() in fname and year in fname:
                # Extract key title words for deeper match
                title_words = re.findall(r'[a-z]{4,}', pdf_name.lower())
                matching_words = sum(1 for w in title_words[:5] if w in fname)
                if matching_words >= 2:
                    return f
    return None


# ── Quality Validation ─────────────────────────────────────────────────────

def validate_markdown(text, pdf_name):
    """Validate converted markdown quality. Returns (is_valid, issues)."""
    issues = []

    if len(text) < MIN_MARKDOWN_LENGTH:
        issues.append(f"too short ({len(text)} chars, min {MIN_MARKDOWN_LENGTH})")

    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        issues.append(f"too few words ({len(words)}, min {MIN_WORD_COUNT})")

    paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 20]
    if len(paragraphs) < MIN_PARAGRAPH_COUNT:
        issues.append(f"too few paragraphs ({len(paragraphs)}, min {MIN_PARAGRAPH_COUNT})")

    # Check for garbled text (high ratio of non-printable or replacement chars)
    if len(text) > 0:
        # Count non-ASCII chars excluding common CJK ranges and accented Latin
        garbled = sum(1 for c in text if ord(c) > 127
                      and not (0xAC00 <= ord(c) <= 0xD7AF)   # Korean
                      and not (0x4E00 <= ord(c) <= 0x9FFF)   # CJK
                      and not (0x00C0 <= ord(c) <= 0x024F)   # Latin Extended
                      and not (0x2000 <= ord(c) <= 0x206F)   # General Punctuation
                      and not (0x2190 <= ord(c) <= 0x21FF)   # Arrows
                      and not (0x2200 <= ord(c) <= 0x22FF)   # Math Operators
                      and c not in "–—''""•·±×÷≈≠≤≥∞∑∫∂√∇αβγδεζηθικλμνξπρστυφχψωΩ")
        ratio = garbled / len(text)
        if ratio > MAX_GARBLED_RATIO:
            issues.append(f"garbled text ratio {ratio:.2%} > {MAX_GARBLED_RATIO:.0%}")

    # Check for signs of scanned/image-only PDF (very short with lots of whitespace)
    if len(text) > 0:
        whitespace_ratio = text.count(' ') / len(text)
        if whitespace_ratio > 0.5 and len(words) < 200:
            issues.append("possibly scanned/image-only PDF")

    is_valid = len(issues) == 0
    return is_valid, issues


# ── Conversion ─────────────────────────────────────────────────────────────

def convert_single_pdf(pdf_path, output_dir, force=False):
    """Convert a single PDF to markdown.

    Returns dict with conversion result.
    """
    pdf_name = pdf_path.name
    result = {
        "pdf": pdf_name,
        "status": "pending",
        "output": None,
        "issues": [],
        "chars": 0,
        "words": 0,
        "time_s": 0,
    }

    # Check if already converted
    if not force:
        existing = find_existing_md(pdf_name, output_dir)
        if existing:
            result["status"] = "skipped_exists"
            result["output"] = str(existing)
            return result

    start = time.time()
    try:
        # Convert with pymupdf4llm
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            show_progress=False,
            page_chunks=False,
        )

        # Validate quality
        is_valid, issues = validate_markdown(md_text, pdf_name)
        result["chars"] = len(md_text)
        result["words"] = len(md_text.split())
        result["issues"] = issues

        if not is_valid:
            result["status"] = "failed_validation"
            log.warning(f"  Validation failed for {pdf_name}: {'; '.join(issues)}")
            # Still save but with a warning prefix
            md_name = pdf_to_md_name(pdf_name)
            out_path = output_dir / md_name
            header = f"<!-- QUALITY WARNING: {'; '.join(issues)} -->\n\n"
            out_path.write_text(header + md_text, encoding="utf-8")
            result["output"] = str(out_path)
            result["status"] = "converted_with_warnings"
        else:
            md_name = pdf_to_md_name(pdf_name)
            out_path = output_dir / md_name
            out_path.write_text(md_text, encoding="utf-8")
            result["output"] = str(out_path)
            result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["issues"] = [str(e)]
        log.error(f"  Error converting {pdf_name}: {e}")

    result["time_s"] = round(time.time() - start, 1)
    return result


def convert_batch(pdf_dir, output_dir, force=False, dry_run=False, max_workers=4):
    """Convert all PDFs in a directory to markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs to convert
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.info("No PDFs found to convert.")
        return []

    log.info(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    # Pre-filter: check which already exist
    to_convert = []
    skipped = 0
    for pdf in pdfs:
        if not force:
            existing = find_existing_md(pdf.name, output_dir)
            if existing:
                skipped += 1
                continue
        to_convert.append(pdf)

    log.info(f"  {skipped} already converted, {len(to_convert)} to convert")

    if dry_run:
        log.info("\nDRY RUN — would convert:")
        for pdf in to_convert:
            log.info(f"  {pdf.name}")
        return []

    # Convert sequentially (pymupdf4llm is not fully thread-safe)
    results = []
    total = len(to_convert)
    for i, pdf in enumerate(to_convert, 1):
        log.info(f"[{i}/{total}] Converting: {pdf.name}")
        result = convert_single_pdf(pdf, output_dir, force=force)
        results.append(result)

        status_emoji = {
            "success": "+",
            "converted_with_warnings": "~",
            "error": "X",
            "skipped_exists": "-",
        }.get(result["status"], "?")

        log.info(f"  [{status_emoji}] {result['status']} "
                 f"({result['chars']} chars, {result['words']} words, {result['time_s']}s)")

    return results


# ── Reporting ──────────────────────────────────────────────────────────────

def print_summary(results):
    """Print conversion summary."""
    if not results:
        return

    success = sum(1 for r in results if r["status"] == "success")
    warnings = sum(1 for r in results if r["status"] == "converted_with_warnings")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    total_chars = sum(r["chars"] for r in results)
    total_words = sum(r["words"] for r in results)
    total_time = sum(r["time_s"] for r in results)

    log.info(f"\n{'='*60}")
    log.info("CONVERSION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Success:     {success}")
    log.info(f"  Warnings:    {warnings}")
    log.info(f"  Errors:      {errors}")
    log.info(f"  Skipped:     {skipped}")
    log.info(f"  Total chars: {total_chars:,}")
    log.info(f"  Total words: {total_words:,}")
    log.info(f"  Total time:  {total_time:.1f}s")

    if errors > 0:
        log.info(f"\nFailed conversions:")
        for r in results:
            if r["status"] == "error":
                log.info(f"  {r['pdf']}: {r['issues']}")

    if warnings > 0:
        log.info(f"\nConversions with quality warnings:")
        for r in results:
            if r["status"] == "converted_with_warnings":
                log.info(f"  {r['pdf']}: {'; '.join(r['issues'])}")


def save_conversion_log(results):
    """Append results to conversion log."""
    existing = []
    if CONVERSION_LOG.exists():
        existing = json.loads(CONVERSION_LOG.read_text(encoding="utf-8"))

    from datetime import datetime
    batch = {
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "results": results,
    }
    existing.append(batch)
    CONVERSION_LOG.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to Markdown Conversion Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be converted without actually converting")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if markdown already exists")
    parser.add_argument("--single", type=str,
                        help="Convert a single PDF file (name or path)")
    parser.add_argument("--recheck", action="store_true",
                        help="Re-validate existing conversions")
    parser.add_argument("--pdf-dir", type=str, default=str(PDF_DIR),
                        help="Source PDF directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output markdown directory")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)

    if args.single:
        # Single file conversion
        pdf_path = Path(args.single)
        if not pdf_path.exists():
            pdf_path = pdf_dir / args.single
        if not pdf_path.exists():
            log.error(f"PDF not found: {args.single}")
            sys.exit(1)

        result = convert_single_pdf(pdf_path, output_dir, force=args.force)
        log.info(f"Result: {result['status']} -> {result.get('output', 'N/A')}")
        if result["issues"]:
            log.info(f"Issues: {'; '.join(result['issues'])}")

    elif args.recheck:
        # Re-validate existing markdown files converted from new_pdf
        log.info("Re-validating existing conversions...")
        issues_found = 0
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            existing = find_existing_md(pdf.name, output_dir)
            if existing:
                text = existing.read_text(encoding="utf-8")
                is_valid, issues = validate_markdown(text, pdf.name)
                if not is_valid:
                    issues_found += 1
                    log.warning(f"  {pdf.name}: {'; '.join(issues)}")
        log.info(f"Recheck complete: {issues_found} files with quality issues")

    else:
        # Batch conversion
        results = convert_batch(
            pdf_dir, output_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
        if results:
            print_summary(results)
            if not args.dry_run:
                save_conversion_log(results)
