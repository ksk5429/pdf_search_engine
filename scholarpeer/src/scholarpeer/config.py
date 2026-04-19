"""Centralized configuration loaded from env vars, .env, and config/default.yaml.

Single source of truth for paths, service URLs, model IDs, and retrieval tuning.
Never read ``os.environ`` outside this module — import :func:`get_settings` instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# SSRF allowlist — external services may only point at these hostnames.
_ALLOWED_OPENALEX_HOSTS = {"api.openalex.org"}
_ALLOWED_S2_HOSTS = {"api.semanticscholar.org"}
_ALLOWED_UNPAYWALL_HOSTS = {"api.unpaywall.org"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT / "config" / "default.yaml"


def _load_yaml_defaults() -> dict[str, object]:
    if not _DEFAULT_YAML.exists():
        return {}
    with _DEFAULT_YAML.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    """Runtime settings. Env vars prefixed ``SP_`` override YAML defaults."""

    model_config = SettingsConfigDict(
        env_prefix="SP_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets — aliased to bare names since they're standard well-known vars.
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="ANTHROPIC_API_KEY",
    )
    polite_email: str = Field(
        default="",
        validation_alias="POLITE_EMAIL",
    )

    # Corpus paths
    corpus_dir: Path = Field(default=Path("F:/TREE_OF_THOUGHT/PHD/papers/literature_review"))
    pdf_dir: Path = Field(default=Path("F:/TREE_OF_THOUGHT/PHD/papers/new_pdf"))

    # Service endpoints
    qdrant_url: str = Field(default="http://localhost:6333")
    grobid_url: str = Field(default="http://localhost:8070")
    openalex_base: str = Field(default="https://api.openalex.org")
    s2_base: str = Field(default="https://api.semanticscholar.org/graph/v1")
    unpaywall_base: str = Field(default="https://api.unpaywall.org/v2")

    # Models
    dense_model: str = Field(default="BAAI/bge-m3")
    dense_dim: int = Field(default=1024)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    colpali_model: str = Field(default="vidore/colqwen2.5-v0.2")
    device: Literal["cuda", "cpu", "mps"] = Field(default="cuda")

    leader_model: str = Field(default="claude-opus-4-7")
    specialist_model: str = Field(default="claude-sonnet-4-6")
    formatter_model: str = Field(default="claude-haiku-4-5")

    # Collection names (must start with sp_)
    collection_dense: str = Field(default="sp_dense")
    collection_sparse: str = Field(default="sp_sparse")
    collection_colpali: str = Field(default="sp_colpali")

    # Retrieval
    top_k_dense: int = Field(default=30)
    top_k_sparse: int = Field(default=30)
    top_k_rerank: int = Field(default=10)
    top_k_visual: int = Field(default=10)
    rrf_k: int = Field(default=60)
    chunk_tokens: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # Agents
    max_turns_leader: int = Field(default=12)
    max_turns_specialist: int = Field(default=6)
    parallel_specialists: bool = Field(default=True)
    self_feedback_rounds: int = Field(default=2)

    # Safety
    require_grounding: bool = Field(default=True)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["console", "json"] = Field(default="console")

    # ── validators ───────────────────────────────────────────────────

    @field_validator("openalex_base")
    @classmethod
    def _validate_openalex(cls, v: str) -> str:
        if urlparse(v).hostname not in _ALLOWED_OPENALEX_HOSTS:
            raise ValueError(f"openalex_base must point at one of {_ALLOWED_OPENALEX_HOSTS}")
        return v

    @field_validator("s2_base")
    @classmethod
    def _validate_s2(cls, v: str) -> str:
        if urlparse(v).hostname not in _ALLOWED_S2_HOSTS:
            raise ValueError(f"s2_base must point at one of {_ALLOWED_S2_HOSTS}")
        return v

    @field_validator("unpaywall_base")
    @classmethod
    def _validate_unpaywall(cls, v: str) -> str:
        if urlparse(v).hostname not in _ALLOWED_UNPAYWALL_HOSTS:
            raise ValueError(f"unpaywall_base must point at one of {_ALLOWED_UNPAYWALL_HOSTS}")
        return v

    @model_validator(mode="after")
    def _validate_chunk_params(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_tokens ({self.chunk_tokens})"
            )
        for name in (
            self.collection_dense,
            self.collection_sparse,
            self.collection_colpali,
        ):
            if not name.startswith("sp_"):
                raise ValueError(f"collection name {name!r} must start with 'sp_'")
        return self

    # Repo paths (derived)
    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @property
    def cache_dir(self) -> Path:
        return _REPO_ROOT / "data" / "cache"

    @property
    def parsed_corpus_dir(self) -> Path:
        return _REPO_ROOT / "data" / "corpus"

    def assert_api_keys(self) -> None:
        """Raise if secrets required at runtime are missing.

        When the Claude Code backend is selected (``SP_LLM_BACKEND=claude_code``,
        the default), ``ANTHROPIC_API_KEY`` is not required — auth flows through
        the local ``claude`` CLI.
        """
        import os

        backend = os.environ.get("SP_LLM_BACKEND", "claude_code").lower()
        if backend == "anthropic":
            key = self.anthropic_api_key.get_secret_value()
            if not key or key.endswith("REPLACE_ME"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Add a real key to scholarpeer/.env "
                    "or switch to SP_LLM_BACKEND=claude_code."
                )
        if not self.polite_email:
            raise RuntimeError("POLITE_EMAIL not set (export or .env).")


@lru_cache(maxsize=1)
def _build_settings() -> Settings:
    # Pydantic Field defaults handle the base case; env vars and .env override.
    # YAML defaults are NOT applied automatically — use the Settings class defaults
    # or set env vars. (config/default.yaml is for reference only.)
    return Settings()


def get_settings() -> Settings:
    """Build ``Settings`` from class defaults + env + .env (env wins)."""
    return _build_settings()


def reset_settings_cache() -> None:
    """Clear the cached ``Settings`` instance. Call from tests after env patching."""
    _build_settings.cache_clear()
