"""Shared prompt scaffolding for specialist agents."""

from __future__ import annotations

from scholarpeer.schemas.retrieval import RetrievalLog

OUTPUT_FORMAT = """\
Respond with ONLY a JSON array. No prose outside the array. Schema:

[
  {
    "severity": "critical" | "major" | "minor" | "suggestion" | "strength",
    "section_ref": "string or null",
    "comment": "one coherent critique or observation (<= 120 words)",
    "evidence_citations": ["SP:<paper_id>", "OA:W...", "S2:..."],
    "confidence": 0.0-1.0
  },
  ...
]

Hard rules:
- Every comment MUST include at least one evidence_citations entry unless severity is "strength".
- Use only citation keys that appear in the provided context. Fabricated keys are a fatal error.
- Keep comments specific (cite section, equation, or figure when possible). Avoid generic advice.
- 3-6 comments per response.
"""


def format_retrieval_context(log: RetrievalLog, max_hits: int = 20) -> str:
    """Render recent retrievals as citation-keyed context blocks."""
    lines = []
    seen: set[str] = set()
    for hit in reversed(log.hits):
        key = f"SP:{hit.chunk.paper_id}"
        if key in seen:
            continue
        seen.add(key)
        section = f" ({hit.chunk.section})" if hit.chunk.section else ""
        lines.append(f"[{key}]{section}\n{hit.chunk.text.strip()}\n")
        if len(seen) >= max_hits:
            break
    return "\n---\n".join(lines) if lines else "(no retrievals yet)"
