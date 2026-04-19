"""Backend-neutral LLM interface.

Every caller passes ``system``, ``user``, ``model``, ``max_tokens`` and gets a
completion string back. Backends translate to their native call shape.
"""

from __future__ import annotations

import os
from typing import Protocol

from scholarpeer.config import get_settings
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Any provider-side failure surfaced back to callers."""


class LLMBackend(Protocol):
    """Minimal completion interface shared by all backends."""

    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> str: ...


# ── model alias translation ─────────────────────────────────────────────────

# Our settings use full model IDs (e.g. ``claude-sonnet-4-6``). The Claude Code
# SDK uses short aliases (``opus``, ``sonnet``, ``haiku``). Map between them so
# callers can share the same config fields regardless of backend.
_ALIAS_TO_SHORT = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "inherit": "inherit",
}


def map_model_alias(model: str) -> str:
    """Translate a model ID to the short alias the Claude Code SDK accepts."""
    if not model:
        return "sonnet"
    if model in _ALIAS_TO_SHORT:
        return _ALIAS_TO_SHORT[model]
    lowered = model.lower()
    if "opus" in lowered:
        return "opus"
    if "haiku" in lowered:
        return "haiku"
    if "sonnet" in lowered:
        return "sonnet"
    return "sonnet"


# ── factory ─────────────────────────────────────────────────────────────────

_CACHED: LLMBackend | None = None


def get_backend(force: str | None = None) -> LLMBackend:
    """Return the configured backend. Defaults to ``claude_code``.

    ``force`` overrides both the env var and the cache. Also respected:
    ``SP_LLM_BACKEND=claude_code|anthropic``.
    """
    global _CACHED
    name = (force or os.environ.get("SP_LLM_BACKEND", "claude_code")).lower()
    if _CACHED is not None and force is None and _CACHED.name == name:
        return _CACHED

    if name == "anthropic":
        from scholarpeer.llm.anthropic_api import AnthropicBackend

        settings = get_settings()
        backend = AnthropicBackend(api_key=settings.anthropic_api_key.get_secret_value())
    elif name in {"claude_code", "cc", "claude-code"}:
        from scholarpeer.llm.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
    else:
        raise LLMError(f"Unknown LLM backend: {name!r} (expected claude_code | anthropic)")

    log.info("llm.backend.select", name=backend.name)
    _CACHED = backend
    return backend


def reset_backend_cache() -> None:
    """Drop the cached backend — useful in tests or when env vars change."""
    global _CACHED
    _CACHED = None
