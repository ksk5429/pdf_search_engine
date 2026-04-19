"""ColPali page indexer — render PDF pages to images, encode with ColQwen2.5,
upsert multi-vectors into a Qdrant collection configured for late-interaction.

Uses **PyMuPDF** (``fitz``) for page rendering so no external poppler binary is
required. Processes pages in bounded batches to fit on consumer GPUs (~12 GB).

Requires:
  - ``pip install -e '.[colpali]'`` (colpali-engine)
  - CUDA GPU (the model is too slow on CPU for any real corpus).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.index.qdrant_client import QdrantStore
from scholarpeer.logging import get_logger
from scholarpeer.utils.hashing import short_hash

log = get_logger(__name__)


@dataclass(frozen=True)
class VisualIndexResult:
    paper_id: str
    pdf_path: Path
    pages_indexed: int
    elapsed_s: float
    skipped: bool = False
    error: str | None = None


class ColPaliIndexer:
    def __init__(
        self,
        store: QdrantStore | None = None,
        model_name: str | None = None,
        device: str | None = None,
        collection: str | None = None,
        page_batch: int = 4,
        max_pages_per_pdf: int = 40,
    ) -> None:
        settings = get_settings()
        self._store = store or QdrantStore()
        self._model_name = model_name or settings.colpali_model
        self._device = device or settings.device
        self._collection = collection or settings.collection_colpali
        self._page_batch = page_batch
        self._max_pages = max_pages_per_pdf
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        log.info("colpali.load", model=self._model_name, device=self._device)
        self._model = ColQwen2_5.from_pretrained(
            self._model_name,
            torch_dtype="float16",
            device_map=self._device,
        ).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(self._model_name)

    # ── paper-level ──────────────────────────────────────────────────────

    def already_indexed(self, paper_id: str) -> int:
        """Return the number of pages already stored for this paper. 0 if none."""
        from qdrant_client.http import models as qm

        self._store._guard(self._collection)  # noqa: SLF001
        existing = {c.name for c in self._store._client.get_collections().collections}  # noqa: SLF001
        if self._collection not in existing:
            return 0
        flt = qm.Filter(
            must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))]
        )
        cnt = self._store._client.count(  # noqa: SLF001
            collection_name=self._collection, count_filter=flt, exact=True
        )
        return int(cnt.count)

    def index_pdf(
        self, pdf_path: Path, paper_id: str, *, dpi: int = 144, skip_existing: bool = True
    ) -> VisualIndexResult:
        """Render each page, encode in batches, and upsert multi-vectors."""
        start = time.time()

        self._store._guard(self._collection)  # noqa: SLF001
        self._store.ensure_colpali_collection(self._collection)

        if skip_existing and self.already_indexed(paper_id) > 0:
            log.info("colpali.skip", paper_id=paper_id, pdf=pdf_path.name)
            return VisualIndexResult(
                paper_id=paper_id,
                pdf_path=pdf_path,
                pages_indexed=0,
                elapsed_s=time.time() - start,
                skipped=True,
            )

        try:
            images = _render_pdf_pymupdf(pdf_path, dpi=dpi, max_pages=self._max_pages)
        except Exception as exc:  # noqa: BLE001
            log.warning("colpali.render_failed", pdf=str(pdf_path), error=str(exc))
            return VisualIndexResult(
                paper_id=paper_id,
                pdf_path=pdf_path,
                pages_indexed=0,
                elapsed_s=time.time() - start,
                error=f"render: {exc}",
            )

        if not images:
            log.warning("colpali.empty_pdf", pdf=str(pdf_path))
            return VisualIndexResult(
                paper_id=paper_id, pdf_path=pdf_path, pages_indexed=0, elapsed_s=time.time() - start
            )

        self._load()
        total_pages = len(images)
        points_all: list = []
        try:
            for batch_start in range(0, total_pages, self._page_batch):
                batch_imgs = images[batch_start : batch_start + self._page_batch]
                points_all.extend(
                    self._encode_and_build_points(batch_imgs, paper_id, pdf_path, batch_start)
                )
        except Exception as exc:  # noqa: BLE001 — keep overall run moving
            log.error("colpali.encode_failed", pdf=str(pdf_path), error=str(exc), exc_info=True)
            return VisualIndexResult(
                paper_id=paper_id,
                pdf_path=pdf_path,
                pages_indexed=0,
                elapsed_s=time.time() - start,
                error=f"encode: {exc}",
            )

        if points_all:
            self._store._client.upsert(  # noqa: SLF001
                collection_name=self._collection, points=points_all, wait=True
            )
        elapsed = time.time() - start
        log.info(
            "colpali.indexed",
            paper_id=paper_id,
            pdf=pdf_path.name,
            pages=len(points_all),
            elapsed_s=round(elapsed, 1),
        )
        return VisualIndexResult(
            paper_id=paper_id,
            pdf_path=pdf_path,
            pages_indexed=len(points_all),
            elapsed_s=elapsed,
        )

    def _encode_and_build_points(
        self, imgs: list, paper_id: str, pdf_path: Path, offset: int
    ) -> list:
        from qdrant_client.http import models as qm
        import torch

        batch = self._processor.process_images(imgs).to(self._model.device)  # type: ignore[union-attr]
        with torch.no_grad():
            embeds = self._model(**batch)  # (B, P, D)
        multivecs = embeds.to(dtype=torch.float32).cpu().numpy().tolist()
        points: list = []
        for local_idx, pv in enumerate(multivecs):
            page_idx = offset + local_idx
            pid = short_hash(f"{paper_id}::page::{page_idx}")
            points.append(
                qm.PointStruct(
                    id=_uuid_from_str(pid),
                    vector={"colpali": pv},
                    payload={
                        "paper_id": paper_id,
                        "page": page_idx + 1,
                        "pdf_path": str(pdf_path),
                    },
                )
            )
        del embeds, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return points

    # ── query-side ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Encode query as multi-vector and search via MAX_SIM late-interaction."""
        import torch

        self._store._guard(self._collection)  # noqa: SLF001
        self._load()
        batch = self._processor.process_queries([query]).to(self._model.device)  # type: ignore[union-attr]
        with torch.no_grad():
            emb = self._model(**batch)  # type: ignore[union-attr]
        qvec = emb[0].to(dtype=torch.float32).cpu().numpy().tolist()

        res = self._store._client.query_points(  # noqa: SLF001
            collection_name=self._collection,
            query=qvec,
            using="colpali",
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "paper_id": p.payload.get("paper_id"),
                "page": p.payload.get("page"),
                "pdf_path": p.payload.get("pdf_path"),
                "score": float(p.score),
            }
            for p in res.points
        ]


# ── helpers ────────────────────────────────────────────────────────────────


def _render_pdf_pymupdf(pdf_path: Path, *, dpi: int = 144, max_pages: int = 40) -> list:
    """Render PDF to PIL images using PyMuPDF. Caps at ``max_pages``."""
    import fitz  # PyMuPDF
    from PIL import Image

    images: list = []
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def _uuid_from_str(s: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_OID, s))
