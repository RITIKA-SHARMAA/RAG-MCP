"""Builds the searchable corpus: walk a directory, chunk it, index it.

The whole index lives in memory and is built once at startup. For a corpus of
documentation this is measured in megabytes, and it removes an entire class of
failure (a stale or unreachable vector store) from a server whose only job is to
answer a tool call quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .chunking import Chunk, chunk_text
from .config import INDEXED_SUFFIXES, Settings
from .tokenizer import tokenize


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be read at all."""


@dataclass
class Document:
    source: str
    path: Path
    text: str
    chunk_count: int = 0


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class Corpus:
    settings: Settings
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    index: BM25Index = field(default_factory=BM25Index)
    skipped: list[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, settings: Settings) -> "Corpus":
        root = settings.corpus_dir
        if not root.exists():
            raise CorpusError(f"corpus directory does not exist: {root}")
        if not root.is_dir():
            raise CorpusError(f"corpus path is not a directory: {root}")

        corpus = cls(settings=settings)
        # Sorted so the index, and therefore every chunk id, is reproducible
        # across machines and runs.
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in INDEXED_SUFFIXES:
                continue
            source = path.relative_to(root).as_posix()
            if path.stat().st_size > settings.max_file_bytes:
                corpus.skipped.append(f"{source} (larger than {settings.max_file_bytes} bytes)")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                corpus.skipped.append(f"{source} (not valid UTF-8)")
                continue
            corpus._add_document(source, path, text)

        return corpus

    def _add_document(self, source: str, path: Path, text: str) -> None:
        chunks = chunk_text(
            text,
            source,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        for chunk in chunks:
            self.chunks.append(chunk)
            self.index.add(tokenize(chunk.text))
        self.documents[source] = Document(
            source=source, path=path, text=text, chunk_count=len(chunks)
        )

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        return [
            SearchHit(chunk=self.chunks[i], score=round(score, 4))
            for i, score in self.index.search(tokens, top_k)
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(chunk_id)

    def get_document(self, source: str) -> Document:
        try:
            return self.documents[source]
        except KeyError as exc:
            raise KeyError(source) from exc
