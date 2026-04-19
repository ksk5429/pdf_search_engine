"""MARG-style specialist agents — novelty, methodology, clarity, reproducibility, related-work."""

from scholarpeer.agents.specialists.clarity import ClaritySpecialist
from scholarpeer.agents.specialists.methodology import MethodologySpecialist
from scholarpeer.agents.specialists.novelty import NoveltySpecialist
from scholarpeer.agents.specialists.related_work import RelatedWorkSpecialist
from scholarpeer.agents.specialists.reproducibility import ReproducibilitySpecialist

__all__ = [
    "ClaritySpecialist",
    "MethodologySpecialist",
    "NoveltySpecialist",
    "RelatedWorkSpecialist",
    "ReproducibilitySpecialist",
]
