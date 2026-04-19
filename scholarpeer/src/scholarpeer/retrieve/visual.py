"""ColPali / ColQwen2.5 visual retrieval.

ColPali encodes rendered page images directly with a VLM + late-interaction matcher,
beating OCR->embed pipelines on ViDoRe V2/V3. This module is optional: requires
``pip install -e '.[colpali]'`` and a CUDA GPU.
"""

from __future__ import annotations

from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class ColPaliRetriever:
    """Wrapper around ``byaldi`` (the reference ColPali wrapper).

    ``byaldi`` builds its own on-disk index; for a unified Qdrant-native flow use
    :class:`scholarpeer.index.qdrant_client.QdrantStore.ensure_colpali_collection`
    and emit multi-vectors directly.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.colpali_model
        self._device = device or settings.device
        self._rag = None

    def _load(self) -> None:
        if self._rag is not None:
            return
        from byaldi import RAGMultiModalModel

        log.info("colpali.load", model=self._model_name, device=self._device)
        self._rag = RAGMultiModalModel.from_pretrained(
            self._model_name,
            device=self._device,
        )

    def index(self, pdf_dir: Path, index_name: str = "sp_colpali") -> None:
        self._load()
        self._rag.index(  # type: ignore[union-attr]
            input_path=str(pdf_dir),
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=False,
        )

    def search(self, query: str, k: int = 10) -> list[dict]:
        self._load()
        results = self._rag.search(query, k=k)  # type: ignore[union-attr]
        return [
            {
                "doc_id": r.doc_id,
                "page": r.page_num,
                "score": float(r.score),
            }
            for r in results
        ]
