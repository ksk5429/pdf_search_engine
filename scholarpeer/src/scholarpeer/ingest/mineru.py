"""MinerU 2.5-Pro wrapper. State-of-the-art PDF -> Markdown + structured JSON.

Requires ``magic-pdf`` installed with model weights (run ``magic-pdf --install-models``
once per machine). Falls back to pymupdf4llm if MinerU is unavailable, so the
pipeline still works on machines without the heavy VLM weights.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import PaperSection

log = get_logger(__name__)


@dataclass(frozen=True)
class MinerUResult:
    markdown: str
    sections: tuple[PaperSection, ...]
    figures: tuple[dict, ...]
    tables: tuple[dict, ...]
    equations: tuple[dict, ...]
    layout_json: dict
    backend: str  # "mineru" | "pymupdf4llm"


class MinerUParser:
    """Parse PDFs to Markdown via MinerU; fall back to pymupdf4llm if missing."""

    def __init__(self, output_dir: Path | None = None) -> None:
        settings = get_settings()
        self._out = output_dir or (settings.cache_dir / "mineru")
        self._out.mkdir(parents=True, exist_ok=True)
        self._backend = self._detect_backend()
        log.info("mineru.backend", backend=self._backend)

    @staticmethod
    def _detect_backend() -> str:
        # MinerU requires both the Python package AND the ``magic-pdf`` CLI to be
        # on PATH. On Windows venvs the CLI lives in ``.venv/Scripts`` which may
        # not be on the caller's PATH, so fall back when either is missing.
        from shutil import which

        try:
            import magic_pdf  # noqa: F401
            if which("magic-pdf"):
                return "mineru"
        except ImportError:
            pass
        try:
            import pymupdf4llm  # noqa: F401

            return "pymupdf4llm"
        except ImportError:
            return "none"

    def parse(self, pdf_path: Path) -> MinerUResult:
        if self._backend == "mineru":
            return self._parse_mineru(pdf_path)
        if self._backend == "pymupdf4llm":
            return self._parse_pymupdf4llm(pdf_path)
        raise RuntimeError(
            "No PDF parser installed. Install MinerU (`pip install 'magic-pdf[full]'`) "
            "or pymupdf4llm."
        )

    def _parse_mineru(self, pdf_path: Path) -> MinerUResult:
        """Call MinerU CLI and parse its output directory."""
        work_dir = self._out / pdf_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "magic-pdf",
            "-p",
            str(pdf_path),
            "-o",
            str(work_dir),
            "-m",
            "auto",
        ]
        log.debug("mineru.cli", cmd=cmd)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            log.warning("mineru.failed", stderr=proc.stderr[:500])
            return self._parse_pymupdf4llm(pdf_path)

        md_path = _find_file(work_dir, ".md")
        layout_path = _find_file(work_dir, ".json")
        markdown = md_path.read_text(encoding="utf-8") if md_path else ""
        layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path else {}

        sections = _sections_from_markdown(markdown)
        figures, tables, equations = _extract_artifacts(layout)
        return MinerUResult(
            markdown=markdown,
            sections=sections,
            figures=figures,
            tables=tables,
            equations=equations,
            layout_json=layout,
            backend="mineru",
        )

    def _parse_pymupdf4llm(self, pdf_path: Path) -> MinerUResult:
        """Fallback — lighter, CPU-only, no figures/tables structure."""
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False, page_chunks=False)
        sections = _sections_from_markdown(md)
        return MinerUResult(
            markdown=md,
            sections=sections,
            figures=(),
            tables=(),
            equations=(),
            layout_json={},
            backend="pymupdf4llm",
        )


# ── helpers ────────────────────────────────────────────────────────────────


def _find_file(root: Path, suffix: str) -> Path | None:
    for p in root.rglob(f"*{suffix}"):
        return p
    return None


def _sections_from_markdown(md: str) -> tuple[PaperSection, ...]:
    """Split markdown into sections by heading. Level inferred from ``#`` count."""
    sections: list[PaperSection] = []
    buffer: list[str] = []
    current_heading = "Preamble"
    current_level = 1
    order = 0

    for line in md.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    sections.append(
                        PaperSection(
                            heading=current_heading,
                            level=current_level,
                            text=text,
                            order=order,
                        )
                    )
                    order += 1
            hashes = len(stripped) - len(stripped.lstrip("#"))
            current_heading = stripped.lstrip("#").strip() or f"Section {order + 1}"
            current_level = min(max(hashes, 1), 6)
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                PaperSection(
                    heading=current_heading,
                    level=current_level,
                    text=text,
                    order=order,
                )
            )

    return tuple(sections)


def _extract_artifacts(layout: dict) -> tuple[tuple[dict, ...], tuple[dict, ...], tuple[dict, ...]]:
    figures, tables, equations = [], [], []
    for item in layout.get("pdf_info", []) or layout.get("blocks", []):
        block_type = item.get("type", "")
        if block_type == "image":
            figures.append(item)
        elif block_type == "table":
            tables.append(item)
        elif block_type in {"equation", "interline_equation"}:
            equations.append(item)
    return tuple(figures), tuple(tables), tuple(equations)
