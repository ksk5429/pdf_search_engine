"""NetworkX-backed graph storage for the LightRAG layer.

Persisted as a GPickle on disk. Fast enough for <1M edges; swap for Neo4j if the
corpus grows beyond that scale.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from scholarpeer.config import get_settings
from scholarpeer.graph.extract import GraphTriple
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class GraphStore:
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self._path = path or (settings.cache_dir / "graph" / "lightrag.pkl")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._graph: nx.MultiDiGraph = self._load()

    def _load(self) -> nx.MultiDiGraph:
        if self._path.exists():
            try:
                with self._path.open("rb") as fh:
                    return pickle.load(fh)  # noqa: S301 — trusted local cache
            except Exception as exc:  # noqa: BLE001
                log.warning("graph.load_failed", error=str(exc))
        return nx.MultiDiGraph()

    def save(self) -> None:
        with self._path.open("wb") as fh:
            pickle.dump(self._graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(
            "graph.saved",
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges(),
        )

    def add_triples(self, triples: Iterable[GraphTriple]) -> int:
        added = 0
        for t in triples:
            self._graph.add_node(t.subject)
            self._graph.add_node(t.object)
            self._graph.add_edge(
                t.subject,
                t.object,
                predicate=t.predicate,
                chunk_id=t.chunk_id,
                paper_id=t.paper_id,
            )
            added += 1
        return added

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def neighbors(self, node: str, hops: int = 1) -> set[str]:
        if node not in self._graph:
            return set()
        frontier: set[str] = {node}
        visited: set[str] = set()
        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier.update(self._graph.successors(n))
                next_frontier.update(self._graph.predecessors(n))
            visited |= frontier
            frontier = next_frontier - visited
        return visited | frontier
