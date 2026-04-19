"""Claude Code backend via ``claude-agent-sdk`` — uses local ``claude`` CLI auth.

Spawns Claude Code as a subprocess and streams the assistant response back.
Inherits whatever auth the local ``claude`` CLI uses (Max / Pro / API key), so
a Max subscription costs nothing per call beyond the subscription itself.

Async under the hood; we present a sync ``complete()`` by wrapping with
``asyncio.run()`` per call. Acceptable for specialist/formatter calls where
invocation volume is low; for the LightRAG graph builder we batch.
"""

from __future__ import annotations

import asyncio
import os

from scholarpeer.llm.base import LLMError, map_model_alias
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(self) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "claude-agent-sdk not installed. Run `pip install claude-agent-sdk`."
            ) from exc

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> str:
        # max_tokens is a hint only — Claude Code manages the internal context.
        return asyncio.run(self._complete_async(system=system, user=user, model=model))

    async def _complete_async(self, *, system: str, user: str, model: str) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=system,
            model=map_model_alias(model),
            # Pure generation: no tools, no working directory interaction.
            allowed_tools=[],
            max_turns=1,
            # Don't pick up project-level CLAUDE.md etc. during these inner calls —
            # the inner prompt is the full instruction.
            setting_sources=[],
            cwd=os.getcwd(),
        )

        parts: list[str] = []
        saw_result = False
        try:
            async for msg in query(prompt=user, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    saw_result = True
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Claude Code query failed: {exc}") from exc

        text = "".join(parts).strip()
        if not text:
            log.warning("claude_code.empty_response", saw_result=saw_result)
        return text
