"""Runtime configuration with environment-variable overrides.

Every setting can be overridden with an environment variable named
``RAG_<FIELD>`` (for example ``RAG_LLM_MODEL=llama3.2:1b``). The Ollama
host is read by the ``ollama`` library itself from ``OLLAMA_HOST``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "RAG_"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".rag_cache")
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"
    chunk_words: int = 120
    chunk_overlap: int = 20
    top_k: int = 3
    min_score: float = 0.15
    llm_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_words")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1 (cosine similarity)")
        if self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        defaults = cls()

        def get(name: str, default, cast=str):
            raw = env.get(ENV_PREFIX + name)
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid value for {ENV_PREFIX}{name}: {raw!r}") from exc

        return cls(
            data_dir=get("DATA_DIR", defaults.data_dir, Path),
            cache_dir=get("CACHE_DIR", defaults.cache_dir, Path),
            embed_model=get("EMBED_MODEL", defaults.embed_model),
            llm_model=get("LLM_MODEL", defaults.llm_model),
            chunk_words=get("CHUNK_WORDS", defaults.chunk_words, int),
            chunk_overlap=get("CHUNK_OVERLAP", defaults.chunk_overlap, int),
            top_k=get("TOP_K", defaults.top_k, int),
            min_score=get("MIN_SCORE", defaults.min_score, float),
            llm_timeout=get("LLM_TIMEOUT", defaults.llm_timeout, float),
        )
