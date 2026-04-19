"""Layer 4 — Reason. Claude Agent SDK orchestration.

The leader agent (Opus) spawns five specialist subagents (Sonnet), each with a MARG-
style role. Every tool call is grounded via ``HybridRetriever`` so citations can be
verified after the fact.
"""

from scholarpeer.agents.base import BaseSpecialist
from scholarpeer.agents.leader import LeaderAgent
from scholarpeer.agents.specialists import (
    ClaritySpecialist,
    MethodologySpecialist,
    NoveltySpecialist,
    RelatedWorkSpecialist,
    ReproducibilitySpecialist,
)
from scholarpeer.agents.tools import build_retrieval_tools

__all__ = [
    "BaseSpecialist",
    "ClaritySpecialist",
    "LeaderAgent",
    "MethodologySpecialist",
    "NoveltySpecialist",
    "RelatedWorkSpecialist",
    "ReproducibilitySpecialist",
    "build_retrieval_tools",
]
