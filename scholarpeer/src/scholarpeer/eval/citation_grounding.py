"""Hard grounding check: every [SOURCE:id] in the review must exist in the RetrievalLog.

This is the backstop on top of the agent-level instructions. If this check fails,
the review is considered contaminated with hallucinations and must be regenerated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scholarpeer.logging import get_logger
from scholarpeer.schemas.retrieval import RetrievalLog
from scholarpeer.schemas.review import Review

log = get_logger(__name__)

_CITE_PAT = re.compile(r"\[(SP|OA|S2|DOI):([A-Za-z0-9._\-]+)\]")


@dataclass(frozen=True)
class CitationGroundingReport:
    total: int
    valid: int
    invalid_citations: tuple[str, ...]

    @property
    def grounded(self) -> bool:
        return not self.invalid_citations

    @property
    def grounding_rate(self) -> float:
        return self.valid / self.total if self.total else 1.0


def verify_grounding(review: Review, log_: RetrievalLog) -> CitationGroundingReport:
    """Confirm every [X:y] in the review appears somewhere in the retrieval log."""
    valid_ids = log_.cited_ids()

    def _extract(text: str) -> list[str]:
        return [f"{src}:{paper_id}" for src, paper_id in _CITE_PAT.findall(text)]

    all_keys: list[str] = []
    for comment in review.comments:
        # Explicit evidence list + any inline [X:y] in the comment body
        all_keys.extend(comment.evidence_citations)
        all_keys.extend(_extract(comment.comment))

    seen: set[str] = set()
    invalid: list[str] = []
    for k in all_keys:
        if k in seen:
            continue
        seen.add(k)
        if k not in valid_ids:
            invalid.append(k)

    report = CitationGroundingReport(
        total=len(seen),
        valid=len(seen) - len(invalid),
        invalid_citations=tuple(invalid),
    )
    if invalid:
        log.warning("grounding.invalid", count=len(invalid), sample=invalid[:5])
    else:
        log.info("grounding.ok", total=report.total)
    return report
