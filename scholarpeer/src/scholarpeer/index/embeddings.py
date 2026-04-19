"""Dense (BGE-M3) and sparse (BM25 via FastEmbed) encoders.

Lazy-loaded: model weights are only downloaded when the first ``.encode()`` is called,
so tests and CLI ``--help`` stay fast.
"""

from __future__ import annotations

from typing import Iterable

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class DenseEmbedder:
    """Sentence-transformer wrapper. Default model: BAAI/bge-m3 (1024-dim, multilingual)."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.dense_model
        self._device = device or settings.device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        log.info("dense.load", model=self._model_name, device=self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        # fp16 on GPU cuts memory ~half and accelerates encode by ~1.5-2x.
        if self._device == "cuda":
            try:
                self._model = self._model.half()
            except Exception:  # noqa: BLE001 — fall back to fp32 if model can't half
                pass

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
    ) -> list[list[float]]:
        self._load()
        texts = list(texts)
        if not texts:
            return []
        # BGE-M3 uses an instruction prefix for queries
        if is_query and "bge" in self._model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class SparseEmbedder:
    """BM25 sparse vectors via Qdrant's FastEmbed BM25 encoder."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from fastembed import SparseTextEmbedding

        log.info("sparse.load", model=self._model_name)
        self._model = SparseTextEmbedding(model_name=self._model_name)

    def encode(self, texts: Iterable[str]) -> list[dict[str, list]]:
        """Return a list of ``{"indices": [...], "values": [...]}`` dicts."""
        self._load()
        texts = list(texts)
        if not texts:
            return []
        out = []
        for sp in self._model.embed(texts):  # type: ignore[union-attr]
            out.append({"indices": list(sp.indices), "values": list(sp.values)})
        return out
