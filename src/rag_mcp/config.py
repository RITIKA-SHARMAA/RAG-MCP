"""Configuration, read from the environment at startup.

An MCP server is launched by its client, usually from a JSON config file, so the
environment is the only configuration surface it reliably has. Everything is
validated here rather than at first use, so a bad setting fails on startup with
a readable message instead of halfway through a tool call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_SIZE = 180
DEFAULT_OVERLAP = 40
DEFAULT_TOP_K = 5
DEFAULT_MAX_FILE_BYTES = 2_000_000
INDEXED_SUFFIXES = (".md", ".txt", ".rst")


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_OVERLAP
    default_top_k: int = DEFAULT_TOP_K
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        corpus = Path(env.get("RAG_CORPUS_DIR", "corpus")).expanduser()
        settings = cls(
            corpus_dir=corpus,
            chunk_size=_positive_int(env, "RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            chunk_overlap=_positive_int(env, "RAG_CHUNK_OVERLAP", DEFAULT_OVERLAP, allow_zero=True),
            default_top_k=_positive_int(env, "RAG_TOP_K", DEFAULT_TOP_K),
            max_file_bytes=_positive_int(env, "RAG_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES),
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({settings.chunk_overlap}) must be smaller than "
                f"RAG_CHUNK_SIZE ({settings.chunk_size})"
            )
        return settings


def _positive_int(env: dict[str, str], key: str, default: int, *, allow_zero: bool = False) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be {'non-negative' if allow_zero else 'positive'}, got {value}")
    return value
