import pytest

from rag_mcp.config import DEFAULT_CHUNK_SIZE, Settings


def test_defaults_apply_when_nothing_is_set():
    settings = Settings.from_env({})
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE
    assert settings.corpus_dir.name == "corpus"


def test_values_are_read_from_the_environment():
    settings = Settings.from_env(
        {"RAG_CORPUS_DIR": "/tmp/docs", "RAG_CHUNK_SIZE": "50", "RAG_CHUNK_OVERLAP": "5"}
    )
    assert str(settings.corpus_dir) == "/tmp/docs"
    assert settings.chunk_size == 50
    assert settings.chunk_overlap == 5


def test_overlap_at_or_above_chunk_size_is_rejected_at_startup():
    with pytest.raises(ValueError, match="must be smaller than"):
        Settings.from_env({"RAG_CHUNK_SIZE": "20", "RAG_CHUNK_OVERLAP": "20"})


def test_zero_overlap_is_allowed_but_zero_chunk_size_is_not():
    assert Settings.from_env({"RAG_CHUNK_OVERLAP": "0"}).chunk_overlap == 0
    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE must be positive"):
        Settings.from_env({"RAG_CHUNK_SIZE": "0"})


def test_non_numeric_values_are_rejected_with_the_offending_key():
    with pytest.raises(ValueError, match="RAG_TOP_K must be an integer"):
        Settings.from_env({"RAG_TOP_K": "lots"})


def test_empty_string_falls_back_to_the_default():
    assert Settings.from_env({"RAG_TOP_K": ""}).default_top_k == 5
