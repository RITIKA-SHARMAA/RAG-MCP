"""Word-window chunking with overlap, carrying character offsets.

A retrieval chunk has two jobs that pull against each other. It has to be small
enough that the passage handed back to the model is mostly relevant, and large
enough that the answer is not split across a boundary. The overlap window is the
compromise: every chunk repeats the tail of the one before it, so a fact sitting
on a boundary appears whole in at least one chunk.

Offsets are kept so a chunk can always be traced back to an exact span of the
source file. Retrieval that cannot cite its source is not much use to an agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage."""

    chunk_id: str
    source: str
    ordinal: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    """Split ``text`` into overlapping windows of ``chunk_size`` words.

    ``overlap`` words are repeated at the start of each chunk after the first.
    Raises ValueError on a configuration that would not terminate.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        # Without this the window would never advance and the loop would run
        # forever on any non-empty document.
        raise ValueError("overlap must be smaller than chunk_size")

    words = list(_WORD.finditer(text))
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    ordinal = 0

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        start_char = window[0].start()
        end_char = window[-1].end()
        chunks.append(
            Chunk(
                chunk_id=f"{source}#{ordinal}",
                source=source,
                ordinal=ordinal,
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )
        ordinal += 1
        if start + chunk_size >= len(words):
            # The window already reached the end of the document. Without this
            # the stepping loop would emit a final chunk that is pure overlap.
            break

    return chunks
