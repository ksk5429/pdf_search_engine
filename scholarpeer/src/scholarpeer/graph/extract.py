"""Entity-relation extraction for the LightRAG layer.

Uses a cheap Haiku call to pull (entity_a, relation, entity_b) triples from each
chunk. Cost: ~$0.15 per document at current pricing, ~6000x cheaper than MS-GraphRAG
per the LightRAG paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from scholarpeer.config import get_settings
from scholarpeer.llm import LLMBackend, get_backend
from scholarpeer.logging import get_logger
from scholarpeer.schemas.retrieval import Chunk

log = get_logger(__name__)

_SYSTEM = """\
You are an entity-relation extractor for scientific text. Given a passage, output a
JSON array of triples. Each triple is [subject, predicate, object] where subject and
object are concepts, methods, datasets, metrics, or authors, and predicate is a
short verb phrase. Return ONLY the JSON array. Limit to <=10 triples per passage.
"""


@dataclass(frozen=True)
class GraphTriple:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    paper_id: str


class EntityRelationExtractor:
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        backend: LLMBackend | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.formatter_model  # Haiku is plenty
        self._max_tokens = max_tokens
        self._backend = backend or get_backend()

    def extract(self, chunk: Chunk) -> list[GraphTriple]:
        raw = self._backend.complete(
            system=_SYSTEM,
            user=chunk.text[:3000],
            model=self._model,
            max_tokens=self._max_tokens,
        )
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            log.debug("graph.extract_empty", chunk_id=chunk.chunk_id)
            return []
        triples: list[GraphTriple] = []
        for item in data:
            if isinstance(item, list) and len(item) == 3:
                s, p, o = item
                if all(isinstance(x, str) and x.strip() for x in (s, p, o)):
                    triples.append(
                        GraphTriple(
                            subject=s.strip().lower()[:80],
                            predicate=p.strip().lower()[:60],
                            object=o.strip().lower()[:80],
                            chunk_id=chunk.chunk_id,
                            paper_id=chunk.paper_id,
                        )
                    )
        return triples
