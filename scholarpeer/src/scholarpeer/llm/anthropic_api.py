"""Direct Anthropic API backend (paid per token)."""

from __future__ import annotations

from anthropic import Anthropic

from scholarpeer.llm.base import LLMError
from scholarpeer.logging import get_logger

log = get_logger(__name__)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        if not api_key or api_key.endswith("REPLACE_ME"):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Put a real key in scholarpeer/.env "
                "or switch to the Claude Code backend (SP_LLM_BACKEND=claude_code)."
            )
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> str:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
