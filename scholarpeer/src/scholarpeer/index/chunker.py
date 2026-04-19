"""Token-aware chunker that respects section boundaries.

Strategy: split by section; for sections larger than ``chunk_tokens``, window with
``chunk_overlap`` tokens of overlap. Uses ``tiktoken`` for token counts so chunk sizes
match LLM context budgets accurately.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from scholarpeer.config import get_settings
from scholarpeer.schemas.paper import Paper
from scholarpeer.schemas.retrieval import Chunk
from scholarpeer.utils.hashing import short_hash

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class _Window:
    text: str
    token_count: int


class SectionChunker:
    def __init__(self, chunk_tokens: int | None = None, overlap: int | None = None) -> None:
        settings = get_settings()
        self._chunk_tokens = chunk_tokens or settings.chunk_tokens
        self._overlap = overlap or settings.chunk_overlap

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        chunks: list[Chunk] = []
        order = 0
        # Ensure the abstract is always its own first chunk — it's cheap and high-signal.
        if paper.abstract:
            chunks.append(self._make_chunk(paper, paper.abstract, "abstract", order))
            order += 1

        for sec in paper.sections:
            for window in self._window(sec.text):
                chunks.append(self._make_chunk(paper, window.text, sec.heading, order))
                order += 1
        return chunks

    def _window(self, text: str) -> list[_Window]:
        tokens = _ENC.encode(text or "")
        if not tokens:
            return []
        if len(tokens) <= self._chunk_tokens:
            return [_Window(text=text, token_count=len(tokens))]
        windows: list[_Window] = []
        step = max(1, self._chunk_tokens - self._overlap)
        for start in range(0, len(tokens), step):
            end = start + self._chunk_tokens
            piece = tokens[start:end]
            if not piece:
                break
            windows.append(_Window(text=_ENC.decode(piece), token_count=len(piece)))
            if end >= len(tokens):
                break
        return windows

    def _make_chunk(self, paper: Paper, text: str, section: str, order: int) -> Chunk:
        chunk_id = short_hash(f"{paper.paper_id}::{order}::{text[:64]}")
        token_count = len(_ENC.encode(text))
        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            text=text,
            section=section,
            token_count=token_count,
            order_in_paper=order,
        )
