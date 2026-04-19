"""Base class for MARG-style specialist agents.

A specialist is a small, context-scoped reviewer agent that takes:
  - a focused target-paper excerpt (its ``focus``)
  - a curated set of retrieved passages
  - a shared retrieval log (for grounding)

and returns a list of ``ReviewerComment`` objects.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass

from scholarpeer.config import get_settings
from scholarpeer.llm import LLMBackend, get_backend
from scholarpeer.logging import get_logger
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import RetrievalLog
from scholarpeer.schemas.review import ReviewerComment, ReviewSeverity, SpecialistRole

log = get_logger(__name__)


@dataclass(frozen=True)
class SpecialistInput:
    target: Paper
    focus: str  # the paper section or excerpt this specialist should zero in on
    retrieval_log: RetrievalLog


class BaseSpecialist(abc.ABC):
    """Abstract base. Subclasses fix the role and supply a prompt template."""

    role: SpecialistRole

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 2048,
        backend: LLMBackend | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.specialist_model
        self._max_tokens = max_tokens
        self._backend = backend or get_backend()

    @abc.abstractmethod
    def system_prompt(self) -> str:
        """Role-specific system prompt (MARG-style instructions)."""

    @abc.abstractmethod
    def user_prompt(self, inp: SpecialistInput) -> str:
        """Role-specific user prompt built from the target paper + retrieved passages."""

    def review(self, inp: SpecialistInput) -> list[ReviewerComment]:
        log.info("specialist.review", role=self.role.value, paper=inp.target.paper_id)
        raw = self._backend.complete(
            system=self.system_prompt(),
            user=self.user_prompt(inp),
            model=self._model,
            max_tokens=self._max_tokens,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> list[ReviewerComment]:
        """Parse the structured JSON block the specialist returns. Resilient to prose."""
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("specialist.parse_failed", role=self.role.value, error=str(exc))
            return []

        comments: list[ReviewerComment] = []
        for item in data:
            try:
                comments.append(
                    ReviewerComment(
                        role=self.role,
                        severity=ReviewSeverity(item.get("severity", "minor")),
                        section_ref=item.get("section_ref"),
                        comment=item["comment"],
                        evidence_citations=tuple(item.get("evidence_citations", ())),
                        confidence=float(item.get("confidence", 0.7)),
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one bad item must not kill the rest
                log.warning("specialist.bad_comment", error=str(exc), item=item)
        return comments
