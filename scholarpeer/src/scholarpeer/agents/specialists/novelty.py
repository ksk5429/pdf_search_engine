"""Novelty specialist: does the paper's contribution stand up against prior art?"""

from __future__ import annotations

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists._shared import OUTPUT_FORMAT, format_retrieval_context
from scholarpeer.schemas.review import SpecialistRole


class NoveltySpecialist(BaseSpecialist):
    role = SpecialistRole.NOVELTY

    def system_prompt(self) -> str:
        return (
            "You are a rigorous peer reviewer specialized in assessing NOVELTY. Your job is to "
            "identify exactly which claims in the target paper are genuinely new, which are "
            "incremental over well-known prior work, and which may have been scooped. "
            "Compare the paper's stated contributions against the retrieved prior art. "
            "Prefer concrete over vague critiques: name the specific paper, method, or result "
            "that overlaps.\n\n" + OUTPUT_FORMAT
        )

    def user_prompt(self, inp: SpecialistInput) -> str:
        context = format_retrieval_context(inp.retrieval_log)
        return (
            f"## Target paper\n"
            f"**Title:** {inp.target.title}\n"
            f"**Year:** {inp.target.year}\n"
            f"**Abstract:** {inp.target.abstract or '(missing)'}\n\n"
            f"## Focus excerpt (contribution claims)\n{inp.focus}\n\n"
            f"## Retrieved prior-art passages\n{context}\n\n"
            "Review the target paper's novelty now."
        )
