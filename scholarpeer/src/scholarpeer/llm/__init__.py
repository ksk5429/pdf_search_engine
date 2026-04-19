"""LLM backend abstraction.

Selects between the Anthropic API (``anthropic.Anthropic``) and the local
Claude Code CLI (via ``claude-agent-sdk``). The Claude Code backend inherits
whatever auth the local ``claude`` CLI has (Max / Pro / API key), so it works
with a Max-plan subscription and no paid API key.

Select via env var ``SP_LLM_BACKEND=claude_code|anthropic`` (default ``claude_code``).
"""

from scholarpeer.llm.base import LLMBackend, LLMError, get_backend, map_model_alias

__all__ = ["LLMBackend", "LLMError", "get_backend", "map_model_alias"]
