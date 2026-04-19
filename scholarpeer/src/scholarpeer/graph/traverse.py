"""Multi-hop traversal utilities. HippoRAG-style personalized PageRank is the strong
version; we implement a lighter weighted-BFS for cost control."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from scholarpeer.graph.store import GraphStore
from scholarpeer.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TraversalHit:
    node: str
    hops: int
    paper_ids: frozenset[str]


class GraphTraverser:
    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def personalized_pagerank(
        self,
        seeds: list[str],
        *,
        alpha: float = 0.85,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Return top-k nodes by PPR from ``seeds``. Undirected view for fairness."""
        g = self._store.graph
        seeds = [s for s in seeds if s in g]
        if not seeds:
            return []
        ug = nx.Graph(g)
        personalization = {n: (1.0 if n in seeds else 0.0) for n in ug}
        total = sum(personalization.values())
        if total == 0:
            return []
        personalization = {k: v / total for k, v in personalization.items()}
        scores = nx.pagerank(ug, alpha=alpha, personalization=personalization)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def multi_hop(self, seeds: list[str], hops: int = 2) -> list[TraversalHit]:
        g = self._store.graph
        hits: list[TraversalHit] = []
        for seed in seeds:
            if seed not in g:
                continue
            frontier: set[str] = {seed}
            visited: set[str] = set()
            for h in range(hops):
                next_frontier: set[str] = set()
                for n in frontier:
                    for neighbor in list(g.successors(n)) + list(g.predecessors(n)):
                        if neighbor not in visited:
                            paper_ids = frozenset(
                                edata.get("paper_id", "")
                                for _, _, edata in g.edges(neighbor, data=True)
                                if edata.get("paper_id")
                            )
                            hits.append(TraversalHit(node=neighbor, hops=h + 1, paper_ids=paper_ids))
                            next_frontier.add(neighbor)
                visited |= frontier
                frontier = next_frontier - visited
        return hits
