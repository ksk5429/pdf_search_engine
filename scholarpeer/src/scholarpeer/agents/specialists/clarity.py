"""Clarity specialist: is the presentation clear, correct, and complete?"""

from __future__ import annotations

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists._shared import OUTPUT_FORMAT, format_retrieval_context
from scholarpeer.schemas.review import SpecialistRole


class ClaritySpecialist(BaseSpecialist):
    role = SpecialistRole.CLARITY

    def system_prompt(self) -> str:
        return (
            "You are a peer reviewer specialized in CLARITY and PRESENTATION. Flag: ambiguous "
            "notation, undefined symbols, figures that don't match text claims, inconsistent "
            "terminology, and arguments that skip necessary intermediate steps. Give concrete "
            "before/after suggestions when possible. Avoid purely stylistic nits.\n\n"
            + OUTPUT_FORMAT
        )

    def user_prompt(self, inp: SpecialistInput) -> str:
        context = format_retrieval_context(inp.retrieval_log, max_hits=5)
        return (
            f"## Target paper presentation\n"
            f"**Title:** {inp.target.title}\n\n"
            f"## Focus excerpt\n{inp.focus}\n\n"
            f"## Brief related context (for terminology checks)\n{context}\n\n"
            "Review the presentation now."
        )
