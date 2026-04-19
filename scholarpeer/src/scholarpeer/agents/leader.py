"""Leader agent — orchestrates the full MARG-style review pipeline.

Flow:
  1. Plan retrieval queries from the target paper's abstract + section headings.
  2. Execute retrieval via ``HybridRetriever`` to populate ``RetrievalLog``.
  3. Dispatch all five specialists (concurrently when ``parallel_specialists=True``).
  4. Run the OpenScholar-style self-feedback loop (optional).
  5. Assemble a ``Review`` artifact and invoke the formatter for final polish.
"""

from __future__ import annotations

import concurrent.futures as cf
import uuid
from typing import Sequence

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists import (
    ClaritySpecialist,
    MethodologySpecialist,
    NoveltySpecialist,
    RelatedWorkSpecialist,
    ReproducibilitySpecialist,
)
from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger
from scholarpeer.retrieve.hybrid import HybridRetriever
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import RetrievalLog, RetrievalQuery
from scholarpeer.schemas.review import Review, ReviewerComment, SpecialistRole
from scholarpeer.synthesize.self_feedback import SelfFeedbackLoop

log = get_logger(__name__)


class LeaderAgent:
    """Top-level orchestrator. Single entry point: ``review(paper)``."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        specialists: Sequence[BaseSpecialist] | None = None,
        self_feedback: SelfFeedbackLoop | None = None,
        enable_self_feedback: bool = True,
    ) -> None:
        settings = get_settings()
        self._retriever = retriever or HybridRetriever()
        self._parallel = settings.parallel_specialists
        self._specialists: tuple[BaseSpecialist, ...] = tuple(
            specialists
            or (
                NoveltySpecialist(),
                MethodologySpecialist(),
                ClaritySpecialist(),
                ReproducibilitySpecialist(),
                RelatedWorkSpecialist(),
            )
        )
        self._self_feedback = (
            self_feedback if self_feedback is not None
            else (SelfFeedbackLoop(retriever=self._retriever) if enable_self_feedback else None)
        )
        self._settings = settings

    # ── main entry ───────────────────────────────────────────────────────

    def review(self, paper: Paper) -> Review:
        session_id = str(uuid.uuid4())
        retrieval_log = RetrievalLog(session_id=session_id)
        log.info("leader.start", session=session_id, paper=paper.paper_id, title=paper.title[:80])

        self._plan_and_retrieve(paper, retrieval_log)

        specialist_inputs = self._build_specialist_inputs(paper, retrieval_log)
        comments = self._dispatch_specialists(specialist_inputs)

        review = Review(
            target_paper_id=paper.paper_id,
            target_title=paper.title,
            summary=self._build_summary(paper, comments),
            comments=comments,
            strengths=[c.comment for c in comments if c.severity.value == "strength"][:5],
            weaknesses=[c.comment for c in comments if c.severity.value in {"critical", "major"}][:10],
            recommendation=self._recommend(comments),
            overall_confidence=self._mean_confidence(comments),
            session_id=session_id,
            model_routing={
                "leader": self._settings.leader_model,
                "specialist": self._settings.specialist_model,
                "formatter": self._settings.formatter_model,
            },
        )

        if self._self_feedback is not None:
            rounds = self._self_feedback.refine(review, retrieval_log)
            log.info("leader.self_feedback", rounds=len(rounds), session=session_id)

        log.info("leader.done", comments=len(comments), session=session_id)
        return review

    # ── stages ───────────────────────────────────────────────────────────

    def _plan_and_retrieve(self, paper: Paper, retrieval_log: RetrievalLog) -> None:
        """Derive retrieval queries from title + abstract + section headings."""
        queries = self._derive_queries(paper)
        for q in queries:
            self._retriever.search(
                RetrievalQuery(query=q, top_k=self._settings.top_k_rerank),
                log_to=retrieval_log,
            )
        log.info("leader.retrieved", hits=len(retrieval_log.hits), queries=len(queries))

    @staticmethod
    def _derive_queries(paper: Paper) -> list[str]:
        seeds = [paper.title]
        if paper.abstract:
            seeds.append(paper.abstract[:300])
        for sec in paper.sections[:8]:
            if 3 < len(sec.heading) < 80:
                seeds.append(f"{paper.title} — {sec.heading}")
        seen: set[str] = set()
        out: list[str] = []
        for q in seeds:
            key = q[:120]
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out[:8]

    def _build_specialist_inputs(
        self, paper: Paper, retrieval_log: RetrievalLog
    ) -> list[tuple[BaseSpecialist, SpecialistInput]]:
        focus_by_role = self._focus_excerpts(paper)
        return [
            (
                spec,
                SpecialistInput(
                    target=paper,
                    focus=focus_by_role.get(spec.role, paper.abstract or paper.title),
                    retrieval_log=retrieval_log,
                ),
            )
            for spec in self._specialists
        ]

    @staticmethod
    def _focus_excerpts(paper: Paper) -> dict[SpecialistRole, str]:
        """Pick the most relevant section for each specialist, by heading keyword."""
        by_keyword = {
            SpecialistRole.NOVELTY: ("contribution", "introduction", "abstract"),
            SpecialistRole.METHODOLOGY: ("method", "approach", "experiment"),
            SpecialistRole.CLARITY: ("result", "discussion", "method"),
            SpecialistRole.REPRODUCIBILITY: ("experiment", "implementation", "result"),
            SpecialistRole.RELATED_WORK: ("related", "background", "literature"),
        }
        out: dict[SpecialistRole, str] = {}
        for role, kws in by_keyword.items():
            picked = None
            for sec in paper.sections:
                heading = sec.heading.lower()
                if any(k in heading for k in kws):
                    picked = sec.text
                    break
            out[role] = (picked or paper.abstract or paper.title)[:4000]
        return out

    def _dispatch_specialists(
        self, inputs: list[tuple[BaseSpecialist, SpecialistInput]]
    ) -> list[ReviewerComment]:
        comments: list[ReviewerComment] = []
        if self._parallel:
            with cf.ThreadPoolExecutor(max_workers=len(inputs)) as pool:
                futures = {pool.submit(spec.review, inp): spec.role for spec, inp in inputs}
                for fut in cf.as_completed(futures):
                    try:
                        comments.extend(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        log.error("specialist.failed", role=futures[fut].value, error=str(exc))
        else:
            for spec, inp in inputs:
                try:
                    comments.extend(spec.review(inp))
                except Exception as exc:  # noqa: BLE001
                    log.error("specialist.failed", role=spec.role.value, error=str(exc))
        return comments

    @staticmethod
    def _build_summary(paper: Paper, comments: list[ReviewerComment]) -> str:
        n_crit = sum(1 for c in comments if c.severity.value == "critical")
        n_major = sum(1 for c in comments if c.severity.value == "major")
        n_minor = sum(1 for c in comments if c.severity.value == "minor")
        n_str = sum(1 for c in comments if c.severity.value == "strength")
        return (
            f"Review of '{paper.title}'. "
            f"{n_crit} critical, {n_major} major, {n_minor} minor concerns; "
            f"{n_str} notable strengths."
        )

    @staticmethod
    def _recommend(comments: list[ReviewerComment]) -> str:
        sev = {c.severity.value for c in comments}
        if "critical" in sev:
            return "reject"
        n_major = sum(1 for c in comments if c.severity.value == "major")
        if n_major >= 3:
            return "major revision"
        if n_major >= 1:
            return "minor revision"
        return "accept"

    @staticmethod
    def _mean_confidence(comments: list[ReviewerComment]) -> float:
        if not comments:
            return 0.0
        return sum(c.confidence for c in comments) / len(comments)
