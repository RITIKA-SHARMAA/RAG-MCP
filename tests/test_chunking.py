import pytest

from rag_mcp.chunking import chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_becomes_one_chunk():
    chunks = chunk_text("a handful of words", "doc.md", chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc.md#0"
    assert chunks[0].text == "a handful of words"


def test_windows_advance_by_size_minus_overlap():
    chunks = chunk_text(words(30), "doc.md", chunk_size=10, overlap=4)
    # step = 6, so windows start at word 0, 6, 12, 18, 24
    assert [c.ordinal for c in chunks] == [0, 1, 2, 3, 4]
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].text.split()[0] == "w6"


def test_consecutive_chunks_share_the_overlap_words():
    chunks = chunk_text(words(30), "doc.md", chunk_size=10, overlap=4)
    tail = chunks[0].text.split()[-4:]
    head = chunks[1].text.split()[:4]
    assert tail == head


def test_offsets_point_back_at_the_exact_source_span():
    text = words(30)
    for chunk in chunk_text(text, "doc.md", chunk_size=10, overlap=4):
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_no_trailing_chunk_that_is_pure_overlap():
    # 20 words, size 10, step 6: windows at 0, 6, 12. A window starting at 18
    # would repeat only what chunk 2 already contains.
    chunks = chunk_text(words(20), "doc.md", chunk_size=10, overlap=4)
    assert len(chunks) == 3
    assert chunks[-1].text.split()[-1] == "w19"


def test_empty_and_whitespace_only_text_produces_no_chunks():
    assert chunk_text("", "doc.md") == []
    assert chunk_text("   \n\t ", "doc.md") == []


def test_overlap_not_smaller_than_chunk_size_is_rejected():
    # Allowing this would make the window never advance, and indexing any
    # non-empty document would hang.
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text(words(10), "doc.md", chunk_size=5, overlap=5)


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text(words(10), "doc.md", chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap must not be negative"):
        chunk_text(words(10), "doc.md", chunk_size=5, overlap=-1)
