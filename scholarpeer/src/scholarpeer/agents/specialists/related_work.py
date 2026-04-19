"""Related-work specialist: does the paper position itself correctly in the literature?"""

from __future__ import annotations

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists._shared import OUTPUT_FORMAT, format_retrieval_context
from scholarpeer.schemas.review import SpecialistRole


class RelatedWorkSpecialist(BaseSpecialist):
    role = SpecialistRole.RELATED_WORK

    def system_prompt(self) -> str:
        return (
            "You are a peer reviewer specialized in RELATED WORK coverage. Identify important "
            "prior works the authors missed, mis-cited, or incorrectly characterized. Use the "
            "retrieved passages to ground every claim. When recommending a missing citation, "
            "include its citation key (SP:/OA:/S2:) from the context.\n\n" + OUTPUT_FORMAT
        )

    def user_prompt(self, inp: SpecialistInput) -> str:
        context = format_retrieval_context(inp.retrieval_log, max_hits=20)
        return (
            f"## Target paper\n"
            f"**Title:** {inp.target.title}\n"
            f"**Year:** {inp.target.year}\n"
            f"**Abstract:** {inp.target.abstract or '(missing)'}\n\n"
            f"## Focus excerpt (related-work section + key claims)\n{inp.focus}\n\n"
            f"## Retrieved candidate prior art\n{context}\n\n"
            "Review related-work coverage now."
        )
