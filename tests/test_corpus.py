import pytest

from rag_mcp.config import Settings
from rag_mcp.corpus import Corpus, CorpusError


def test_indexes_only_text_suffixes_and_walks_subdirectories(corpus):
    sources = set(corpus.documents)
    assert sources == {"caching.md", "locking.md", "nested/queues.md"}


def test_chunk_ids_are_namespaced_by_source(corpus):
    assert all(chunk.chunk_id.startswith(chunk.source + "#") for chunk in corpus.chunks)


def test_search_finds_the_document_that_covers_the_topic(corpus):
    hits = corpus.search("pessimistic locking version column", top_k=3)
    assert hits
    assert hits[0].chunk.source == "locking.md"


def test_search_returns_nothing_for_a_term_absent_from_the_corpus(corpus):
    assert corpus.search("kubernetes", top_k=3) == []


def test_search_on_stopwords_only_returns_nothing(corpus):
    # Every token is dropped, so there is no query left. Returning the whole
    # corpus here would be worse than returning nothing.
    assert corpus.search("the and of", top_k=3) == []


def test_get_chunk_round_trips(corpus):
    chunk = corpus.chunks[0]
    assert corpus.get_chunk(chunk.chunk_id) is chunk


def test_get_chunk_raises_on_unknown_id(corpus):
    with pytest.raises(KeyError):
        corpus.get_chunk("caching.md#999")


def test_get_document_raises_on_unknown_source(corpus):
    with pytest.raises(KeyError):
        corpus.get_document("missing.md")


def test_oversized_files_are_skipped_and_reported(corpus_dir):
    (corpus_dir / "huge.md").write_text("word " * 5000, encoding="utf-8")
    corpus = Corpus.build(
        Settings(corpus_dir=corpus_dir, chunk_size=40, chunk_overlap=10, max_file_bytes=500)
    )
    assert "huge.md" not in corpus.documents
    assert any(entry.startswith("huge.md") for entry in corpus.skipped)


def test_files_that_are_not_utf8_are_skipped_not_fatal(corpus_dir):
    (corpus_dir / "latin.md").write_bytes(b"\xff\xfe invalid utf-8 bytes")
    corpus = Corpus.build(Settings(corpus_dir=corpus_dir, chunk_size=40, chunk_overlap=10))
    assert "latin.md" not in corpus.documents
    assert any("latin.md" in entry for entry in corpus.skipped)
    assert corpus.documents  # the rest of the corpus still indexed


def test_missing_corpus_directory_fails_loudly(tmp_path):
    with pytest.raises(CorpusError, match="does not exist"):
        Corpus.build(Settings(corpus_dir=tmp_path / "nope"))


def test_corpus_path_that_is_a_file_fails_loudly(tmp_path):
    path = tmp_path / "a-file.md"
    path.write_text("content", encoding="utf-8")
    with pytest.raises(CorpusError, match="not a directory"):
        Corpus.build(Settings(corpus_dir=path))


def test_index_is_reproducible_across_builds(settings):
    first = [c.chunk_id for c in Corpus.build(settings).chunks]
    second = [c.chunk_id for c in Corpus.build(settings).chunks]
    assert first == second
