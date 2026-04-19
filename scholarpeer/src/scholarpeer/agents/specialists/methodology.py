"""Methodology specialist: are experiments sound and sufficient?"""

from __future__ import annotations

from scholarpeer.agents.base import BaseSpecialist, SpecialistInput
from scholarpeer.agents.specialists._shared import OUTPUT_FORMAT, format_retrieval_context
from scholarpeer.schemas.review import SpecialistRole


class MethodologySpecialist(BaseSpecialist):
    role = SpecialistRole.METHODOLOGY

    def system_prompt(self) -> str:
        return (
            "You are a peer reviewer specialized in EXPERIMENTAL METHODOLOGY. Evaluate whether "
            "the methods section supports the paper's claims: datasets, baselines, metrics, "
            "statistical treatment, ablations, and threats to validity. Flag missing controls, "
            "improper baselines, under-specified hyperparameters, and inappropriate statistics. "
            "Where the retrieved literature uses a stronger baseline or protocol, cite it.\n\n"
            + OUTPUT_FORMAT
        )

    def user_prompt(self, inp: SpecialistInput) -> str:
        context = format_retrieval_context(inp.retrieval_log)
        return (
            f"## Target paper methods\n"
            f"**Title:** {inp.target.title}\n"
            f"**Year:** {inp.target.year}\n\n"
            f"## Focus excerpt (methods/experiments)\n{inp.focus}\n\n"
            f"## Retrieved methodological context\n{context}\n\n"
            "Review the methodology now."
        )
