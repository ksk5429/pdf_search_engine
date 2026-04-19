"""Reproducibility specialist: could a competent reader replicate the results?"""

from __future__ import annotations

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists._shared import OUTPUT_FORMAT, format_retrieval_context
from scholarpeer.schemas.review import SpecialistRole


class ReproducibilitySpecialist(BaseSpecialist):
    role = SpecialistRole.REPRODUCIBILITY

    def system_prompt(self) -> str:
        return (
            "You are a peer reviewer specialized in REPRODUCIBILITY. Assess: data availability, "
            "code release, environment specification, random-seed handling, hardware "
            "constraints, and whether reported metrics include confidence intervals / seeds. "
            "Where applicable, compare against community reproducibility norms (ML reproducibility "
            "checklist, ACM artifact evaluation, FAIR principles).\n\n" + OUTPUT_FORMAT
        )

    def user_prompt(self, inp: SpecialistInput) -> str:
        context = format_retrieval_context(inp.retrieval_log, max_hits=8)
        return (
            f"## Target paper\n"
            f"**Title:** {inp.target.title}\n\n"
            f"## Focus excerpt (experimental setup + results)\n{inp.focus}\n\n"
            f"## Retrieved reproducibility context\n{context}\n\n"
            "Review reproducibility now."
        )
