"""Qdrant wrapper — create collections, upsert chunks, search.

Enforces the ``sp_`` prefix invariant on all collections so we never touch collections
created by other projects in the same Qdrant instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger
from scholarpeer.schemas.retrieval import Chunk

log = get_logger(__name__)


@dataclass(frozen=True)
class QdrantUpsert:
    dense: list[list[float]] | None
    sparse: list[dict[str, list]] | None
    chunks: list[Chunk]


class QdrantStore:
    """Thin Qdrant wrapper for ScholarPeer collections.

    Supports two backends:
      * Server mode (default): HTTP client against a running Qdrant service.
      * Local mode: embedded Qdrant storing data on disk at ``path``. Activated
        when ``url`` begins with ``file://`` or ``path`` is supplied. Useful when
        Docker is unavailable.
    """

    def __init__(self, url: str | None = None, path: str | None = None) -> None:
        settings = get_settings()
        self._url = url or settings.qdrant_url
        if path is not None or self._url.startswith("file://"):
            local_path = path or self._url.removeprefix("file://")
            self._client = QdrantClient(path=local_path)
            log.info("qdrant.local", path=local_path)
        else:
            self._client = QdrantClient(url=self._url, timeout=60.0)
            log.info("qdrant.server", url=self._url)
        self._dense_dim = settings.dense_dim
        self._collection_dense = settings.collection_dense
        self._collection_sparse = settings.collection_sparse
        self._collection_colpali = settings.collection_colpali

    # ── guards ───────────────────────────────────────────────────────────

    @staticmethod
    def _guard(name: str) -> None:
        if not name.startswith("sp_"):
            raise ValueError(f"Collection name must start with 'sp_': {name!r}")

    # ── lifecycle ────────────────────────────────────────────────────────

    def ensure_hybrid_collection(self, name: str | None = None) -> None:
        """Create (or confirm) a collection with dense + sparse named vectors."""
        name = name or self._collection_dense
        self._guard(name)

        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            log.debug("qdrant.collection.exists", name=name)
            return

        log.info("qdrant.collection.create", name=name, dim=self._dense_dim)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": qm.VectorParams(
                    size=self._dense_dim,
                    distance=qm.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                "bm25": qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                    modifier=qm.Modifier.IDF,
                )
            },
            quantization_config=qm.BinaryQuantization(
                binary=qm.BinaryQuantizationConfig(always_ram=True),
            ),
        )
        # Useful payload indices for metadata filtering
        self._client.create_payload_index(
            collection_name=name, field_name="paper_id", field_schema=qm.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            collection_name=name, field_name="section", field_schema=qm.PayloadSchemaType.KEYWORD
        )

    def ensure_colpali_collection(self, name: str | None = None, multivec_dim: int = 128) -> None:
        """ColPali late-interaction multi-vector collection."""
        name = name or self._collection_colpali
        self._guard(name)
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return
        log.info("qdrant.colpali.create", name=name)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "colpali": qm.VectorParams(
                    size=multivec_dim,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                    on_disk=True,
                )
            },
        )

    # ── upsert ───────────────────────────────────────────────────────────

    def upsert_chunks(self, collection: str, upsert: QdrantUpsert) -> int:
        self._guard(collection)
        if not upsert.chunks:
            return 0
        n = len(upsert.chunks)
        if upsert.dense is not None and len(upsert.dense) != n:
            raise ValueError(
                f"dense vector count mismatch: got {len(upsert.dense)}, expected {n}"
            )
        if upsert.sparse is not None and len(upsert.sparse) != n:
            raise ValueError(
                f"sparse vector count mismatch: got {len(upsert.sparse)}, expected {n}"
            )
        points: list[qm.PointStruct] = []
        for i, chunk in enumerate(upsert.chunks):
            vector: dict[str, Any] = {}
            if upsert.dense:
                vector["dense"] = upsert.dense[i]
            if upsert.sparse:
                s = upsert.sparse[i]
                vector["bm25"] = qm.SparseVector(indices=s["indices"], values=s["values"])
            points.append(
                qm.PointStruct(
                    id=_uuid_from_str(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "paper_id": chunk.paper_id,
                        "text": chunk.text,
                        "section": chunk.section,
                        "token_count": chunk.token_count,
                        "order_in_paper": chunk.order_in_paper,
                    },
                )
            )
        self._client.upsert(collection_name=collection, points=points, wait=True)
        return len(points)

    # ── search ───────────────────────────────────────────────────────────

    def search_dense(
        self,
        collection: str,
        query_vec: list[float],
        top_k: int = 30,
        filter_paper_ids: tuple[str, ...] = (),
    ) -> list[qm.ScoredPoint]:
        self._guard(collection)
        flt = _filter_paper_ids(filter_paper_ids)
        res = self._client.query_points(
            collection_name=collection,
            query=query_vec,
            using="dense",
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return list(res.points)

    def search_sparse(
        self,
        collection: str,
        sparse_vec: dict[str, list],
        top_k: int = 30,
        filter_paper_ids: tuple[str, ...] = (),
    ) -> list[qm.ScoredPoint]:
        self._guard(collection)
        flt = _filter_paper_ids(filter_paper_ids)
        res = self._client.query_points(
            collection_name=collection,
            query=qm.SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using="bm25",
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return list(res.points)


def _filter_paper_ids(paper_ids: tuple[str, ...]) -> qm.Filter | None:
    if not paper_ids:
        return None
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="paper_id",
                match=qm.MatchAny(any=list(paper_ids)),
            )
        ]
    )


def _uuid_from_str(s: str) -> str:
    """Qdrant requires UUIDs or uint64 IDs. Hash a 12-char chunk id -> UUID5."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_OID, s))
