"""Final formatter — Haiku polishes the review into a human-readable Markdown report.

Hard rule: the formatter must not introduce new citations. It only rearranges and
prose-polishes the specialists' grounded comments.
"""

from __future__ import annotations

from scholarpeer.config import get_settings
from scholarpeer.llm import LLMBackend, get_backend
from scholarpeer.logging import get_logger
from scholarpeer.schemas.review import Review

log = get_logger(__name__)

_SYSTEM = """\
You format peer reviews as clean Markdown. Preserve every citation key verbatim
(keys look like [SP:abc], [OA:W123], [S2:abc]). Do NOT introduce new citations.
Do NOT change the substance of any comment. Output a single Markdown document with:

## Summary
## Strengths
## Weaknesses
## Detailed Comments
  - by specialist role, each comment as a bullet with its severity and citation keys

## Recommendation
"""


class ReviewFormatter:
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        backend: LLMBackend | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.formatter_model
        self._max_tokens = max_tokens
        self._backend = backend or get_backend()

    def format_markdown(self, review: Review) -> str:
        payload = review.model_dump_json(indent=2)
        md = self._backend.complete(
            system=_SYSTEM,
            user=payload,
            model=self._model,
            max_tokens=self._max_tokens,
        ).strip()
        log.info("formatter.done", chars=len(md))
        return md
