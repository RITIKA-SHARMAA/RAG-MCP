from __future__ import annotations

import pytest

from rag_mcp.config import Settings
from rag_mcp.corpus import Corpus

CACHE_DOC = """
# Caching

A read-through cache sits in front of the database. On a miss the service loads
the row, writes it to the cache and returns it. The cache is never populated by
a separate job, so it cannot drift from the database in the way a write-behind
cache can.
"""

LOCKING_DOC = """
# Locking

Pessimistic locking takes the row lock before reading, so a second transaction
waits. Optimistic locking reads first and detects the conflict at write time by
comparing a version column. Pessimistic locking suits high contention on a small
number of rows, such as seats for one event.
"""

QUEUE_DOC = """
# Queues

A task queue decouples the request from the work. The request records the job
and returns, a worker picks it up, and the client polls for the result. Retries
belong to the worker, not the request, and a capped retry count keeps a poison
message from being retried forever.
"""


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "caching.md").write_text(CACHE_DOC, encoding="utf-8")
    (tmp_path / "locking.md").write_text(LOCKING_DOC, encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "queues.md").write_text(QUEUE_DOC, encoding="utf-8")
    # Neither of these should be indexed.
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4 not indexed")
    (tmp_path / "image.png").write_bytes(b"\x89PNG not indexed")
    return tmp_path


@pytest.fixture
def settings(corpus_dir):
    return Settings(corpus_dir=corpus_dir, chunk_size=40, chunk_overlap=10, default_top_k=3)


@pytest.fixture
def corpus(settings):
    return Corpus.build(settings)
