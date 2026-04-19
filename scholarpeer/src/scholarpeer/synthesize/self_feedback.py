"""OpenScholar-style iterative self-feedback loop.

Given a draft review, the loop:
  1. Asks a critic model (Sonnet) to list concrete gaps the draft fails to address.
  2. Issues targeted retrieval calls for each gap (strictly grounded).
  3. Hands the enriched context back to the specialists for a refinement pass.

Loop count is bounded by ``Settings.self_feedback_rounds`` to avoid context rot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from scholarpeer.config import get_settings
from scholarpeer.llm import LLMBackend, get_backend
from scholarpeer.logging import get_logger
from scholarpeer.retrieve.hybrid import HybridRetriever
from scholarpeer.schemas.retrieval import RetrievalLog, RetrievalQuery
from scholarpeer.schemas.review import Review

log = get_logger(__name__)

_CRITIC_SYSTEM = """\
You are a meta-reviewer. Given a draft review, identify up to 5 concrete gaps: claims
that lack citations, missing comparisons to specific prior work, under-specified
evaluation criteria, or absent reproducibility checks. Output a JSON array of strings,
each a targeted retrieval query that would fill that gap. Return ONLY the array.
"""


@dataclass(frozen=True)
class FeedbackRound:
    round_idx: int
    critic_queries: tuple[str, ...]
    new_hits: int


class SelfFeedbackLoop:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        critic_model: str | None = None,
        max_rounds: int | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        settings = get_settings()
        self._retriever = retriever or HybridRetriever()
        self._critic = critic_model or settings.specialist_model
        self._rounds = max_rounds if max_rounds is not None else settings.self_feedback_rounds
        self._backend = backend or get_backend()

    def refine(self, review: Review, retrieval_log: RetrievalLog) -> list[FeedbackRound]:
        """Run up to ``self._rounds`` feedback rounds. Mutates ``retrieval_log`` in place."""
        rounds: list[FeedbackRound] = []
        for i in range(self._rounds):
            queries = self._critique(review)
            if not queries:
                break
            before = len(retrieval_log.hits)
            for q in queries:
                self._retriever.search(RetrievalQuery(query=q, top_k=5), log_to=retrieval_log)
            rounds.append(
                FeedbackRound(
                    round_idx=i,
                    critic_queries=tuple(queries),
                    new_hits=len(retrieval_log.hits) - before,
                )
            )
            log.info("self_feedback.round", round=i, queries=len(queries))
        return rounds

    def _critique(self, review: Review) -> list[str]:
        payload = {
            "title": review.target_title,
            "summary": review.summary,
            "comments": [
                {"role": c.role.value, "severity": c.severity.value, "comment": c.comment}
                for c in review.comments
            ],
        }
        raw = self._backend.complete(
            system=_CRITIC_SYSTEM,
            user=json.dumps(payload),
            model=self._critic,
            max_tokens=800,
        )
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
        return [q for q in data if isinstance(q, str) and 5 < len(q) < 200][:5]
