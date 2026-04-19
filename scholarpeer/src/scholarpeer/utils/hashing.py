"""Deterministic hashing utilities. Used for stable paper / chunk IDs."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def content_sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_hash(data: str | bytes, length: int = 12) -> str:
    """Short prefix of sha256, used for citation IDs like ``SP:abc123def456``."""
    return content_sha256(data)[:length]


def paper_id_from_text(title: str, doi: str | None = None) -> str:
    """Stable 12-char paper ID: DOI if present, else normalized-title hash.

    Policy: paper IDs MUST be reproducible across runs so citations remain stable
    when the corpus is re-ingested.
    """
    if doi:
        return short_hash(doi.lower().strip())
    clean = _NON_ALNUM.sub("", title.lower().strip())
    clean = _WS.sub(" ", clean)
    return short_hash(clean)
