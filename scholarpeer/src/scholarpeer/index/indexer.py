"""End-to-end corpus indexer: ``Paper`` iterable -> Qdrant collection."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from scholarpeer.config import get_settings
from scholarpeer.index.chunker import SectionChunker
from scholarpeer.index.embeddings import DenseEmbedder, SparseEmbedder
from scholarpeer.index.qdrant_client import QdrantStore, QdrantUpsert
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import Chunk

log = get_logger(__name__)


class CorpusIndexer:
    """Chunk, embed (dense + sparse), and upsert every paper into Qdrant."""

    def __init__(
        self,
        store: QdrantStore | None = None,
        chunker: SectionChunker | None = None,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
        collection: str | None = None,
        dense_batch: int = 128,
    ) -> None:
        settings = get_settings()
        self._store = store or QdrantStore()
        self._chunker = chunker or SectionChunker()
        self._dense = dense or DenseEmbedder()
        self._sparse = sparse or SparseEmbedder()
        self._collection = collection or settings.collection_dense
        self._dense_batch = dense_batch

    def index(self, papers: Iterable[Paper]) -> dict[str, int]:
        """Index a batch of papers. Returns ``{paper_id: chunks_upserted}``."""
        self._store.ensure_hybrid_collection(self._collection)
        papers = list(papers)
        total = len(papers)
        stats: dict[str, int] = {}
        start = time.time()
        for i, paper in enumerate(papers, 1):
            stats[paper.paper_id] = self.index_one(paper)
            if i == 1 or i % 25 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                log.info(
                    "index.progress",
                    done=i,
                    total=total,
                    chunks=sum(stats.values()),
                    rate_per_s=round(rate, 2),
                )
        log.info("index.done", papers=total, chunks=sum(stats.values()))
        return stats

    def index_one(self, paper: Paper, max_chunks: int = 200) -> int:
        chunks: list[Chunk] = self._chunker.chunk_paper(paper)
        if not chunks:
            log.warning("index.empty", paper_id=paper.paper_id, title=paper.title[:80])
            return 0
        if len(chunks) > max_chunks:
            log.info("index.cap", paper_id=paper.paper_id, original=len(chunks), capped=max_chunks)
            chunks = chunks[:max_chunks]

        texts = [c.text for c in chunks]
        dense_vecs: list[list[float]] = []
        for start in range(0, len(texts), self._dense_batch):
            dense_vecs.extend(
                self._dense.encode(texts[start : start + self._dense_batch], is_query=False)
            )
        sparse_vecs = self._sparse.encode(texts)

        if len(dense_vecs) != len(chunks) or len(sparse_vecs) != len(chunks):
            raise RuntimeError(
                f"embedding count mismatch: dense={len(dense_vecs)} "
                f"sparse={len(sparse_vecs)} chunks={len(chunks)}"
            )
        upsert = QdrantUpsert(dense=dense_vecs, sparse=sparse_vecs, chunks=chunks)
        n = self._store.upsert_chunks(self._collection, upsert)
        log.debug("index.paper.done", paper_id=paper.paper_id, chunks=n)
        return n


def load_papers_from_corpus(corpus_dir: Path) -> list[Paper]:
    """Load ingested ``Paper`` objects from ``data/corpus/*.json``."""
    papers: list[Paper] = []
    for json_path in sorted(corpus_dir.glob("*.json")):
        if json_path.name.endswith(".sections.json"):
            continue
        try:
            paper = Paper.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("corpus.load.failed", file=str(json_path), error=str(exc))
            continue
        # Re-attach sections from the sidecar file
        sec_path = json_path.with_suffix(".sections.json")
        if sec_path.exists():
            import json as _json

            sec_data = _json.loads(sec_path.read_text(encoding="utf-8"))
            from scholarpeer.schemas.paper import PaperSection

            sections = tuple(PaperSection.model_validate(s) for s in sec_data)
            paper = paper.model_copy(update={"sections": sections})
        papers.append(paper)
    return papers
