"""ColPali page indexer — render PDF pages to images, encode with ColQwen2.5,
upsert multi-vectors into a Qdrant collection configured for late-interaction.

Requires:
  - ``pip install -e '.[colpali]'`` (colpali-engine, byaldi)
  - CUDA GPU (the model is too slow on CPU for any real corpus).
"""

from __future__ import annotations

from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.index.qdrant_client import QdrantStore
from scholarpeer.logging import get_logger
from scholarpeer.utils.hashing import short_hash

log = get_logger(__name__)


class ColPaliIndexer:
    def __init__(
        self,
        store: QdrantStore | None = None,
        model_name: str | None = None,
        device: str | None = None,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or QdrantStore()
        self._model_name = model_name or settings.colpali_model
        self._device = device or settings.device
        self._collection = collection or settings.collection_colpali
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        log.info("colpali_indexer.load", model=self._model_name)
        self._model = ColQwen2_5.from_pretrained(
            self._model_name,
            torch_dtype="float16",
            device_map=self._device,
        ).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(self._model_name)

    def index_pdf(self, pdf_path: Path, paper_id: str, dpi: int = 144) -> int:
        """Render each page, encode, and upsert as a multi-vector point."""
        from pdf2image import convert_from_path
        from qdrant_client.http import models as qm

        self._store._guard(self._collection)  # noqa: SLF001 — explicit invariant check
        self._load()
        self._store.ensure_colpali_collection(self._collection)

        images = convert_from_path(str(pdf_path), dpi=dpi)
        if not images:
            log.warning("colpali.empty_pdf", pdf=str(pdf_path))
            return 0

        batch = self._processor.process_images(images).to(self._model.device)  # type: ignore[union-attr]
        import torch

        with torch.no_grad():
            embeds = self._model(**batch)  # (num_pages, num_patches, dim)
        multivecs = embeds.to(dtype=torch.float32).cpu().numpy().tolist()

        points = []
        for page_idx, pv in enumerate(multivecs):
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
        self._store._client.upsert(  # pylint: disable=protected-access
            collection_name=self._collection, points=points, wait=True
        )
        log.info("colpali.indexed", paper_id=paper_id, pages=len(points))
        return len(points)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Encode query as multi-vector and search via MAX_SIM late-interaction."""
        import torch

        self._store._guard(self._collection)  # noqa: SLF001
        self._load()
        batch = self._processor.process_queries([query]).to(self._model.device)  # type: ignore[union-attr]
        with torch.no_grad():
            emb = self._model(**batch)  # type: ignore[union-attr]
        qvec = emb[0].to(dtype=torch.float32).cpu().numpy().tolist()

        res = self._store._client.query_points(  # pylint: disable=protected-access
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


def _uuid_from_str(s: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_OID, s))
