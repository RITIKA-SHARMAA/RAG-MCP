import json

import pytest

from rag_mcp.tools import ToolError, dispatch, tool_definitions


def call(corpus, name, **arguments):
    content = dispatch(corpus, name, arguments)
    assert len(content) == 1
    assert content[0].type == "text"
    return json.loads(content[0].text)


def test_every_tool_declares_a_description_and_an_object_schema():
    for tool in tool_definitions(5):
        assert tool.description and len(tool.description) > 40
        assert tool.input_schema["type"] == "object"


def test_search_returns_ranked_results_with_source_and_snippet(corpus):
    payload = call(corpus, "search_documents", query="read-through cache")
    assert payload["result_count"] >= 1
    top = payload["results"][0]
    assert top["source"] == "caching.md"
    assert top["chunk_id"].startswith("caching.md#")
    assert top["score"] > 0
    assert "cache" in top["snippet"].lower()


def test_search_honours_top_k(corpus):
    payload = call(corpus, "search_documents", query="the cache lock queue retries", top_k=2)
    assert len(payload["results"]) <= 2


def test_search_with_no_match_says_so_rather_than_returning_nothing_silently(corpus):
    payload = call(corpus, "search_documents", query="kubernetes")
    assert payload["results"] == []
    assert payload["note"]


def test_search_rejects_an_empty_query(corpus):
    with pytest.raises(ToolError, match="non-empty string"):
        dispatch(corpus, "search_documents", {"query": "   "})


def test_search_rejects_out_of_range_and_non_integer_top_k(corpus):
    with pytest.raises(ToolError, match="between 1 and 25"):
        dispatch(corpus, "search_documents", {"query": "cache", "top_k": 0})
    with pytest.raises(ToolError, match="between 1 and 25"):
        dispatch(corpus, "search_documents", {"query": "cache", "top_k": 99})
    with pytest.raises(ToolError, match="must be an integer"):
        dispatch(corpus, "search_documents", {"query": "cache", "top_k": "five"})
    with pytest.raises(ToolError, match="must be an integer"):
        # bool is an int subclass in Python, so this has to be excluded on purpose.
        dispatch(corpus, "search_documents", {"query": "cache", "top_k": True})


def test_get_chunk_returns_full_text_and_offsets_for_a_search_result(corpus):
    found = call(corpus, "search_documents", query="pessimistic locking")
    chunk_id = found["results"][0]["chunk_id"]
    payload = call(corpus, "get_chunk", chunk_id=chunk_id)
    assert payload["chunk_id"] == chunk_id
    assert payload["end_char"] > payload["start_char"]
    assert payload["text"]


def test_get_chunk_reports_an_unknown_id_clearly(corpus):
    with pytest.raises(ToolError, match="no such chunk"):
        dispatch(corpus, "get_chunk", {"chunk_id": "caching.md#404"})


def test_get_document_returns_the_whole_file(corpus):
    payload = call(corpus, "get_document", source="locking.md")
    assert payload["truncated"] is False
    assert "Optimistic locking" in payload["text"]
    assert payload["chunk_count"] >= 1


def test_get_document_reports_an_unknown_source_clearly(corpus):
    with pytest.raises(ToolError, match="no such document"):
        dispatch(corpus, "get_document", {"source": "nope.md"})


def test_get_document_flags_truncation(corpus, monkeypatch):
    monkeypatch.setattr("rag_mcp.tools.MAX_DOCUMENT_CHARS", 20)
    payload = call(corpus, "get_document", source="locking.md")
    assert payload["truncated"] is True
    assert len(payload["text"]) == 20


def test_list_documents_reports_the_whole_index(corpus):
    payload = call(corpus, "list_documents")
    assert payload["document_count"] == 3
    assert payload["chunk_count"] == len(corpus.chunks)
    assert [d["source"] for d in payload["documents"]] == [
        "caching.md",
        "locking.md",
        "nested/queues.md",
    ]


def test_unknown_tool_is_rejected(corpus):
    with pytest.raises(ToolError, match="unknown tool"):
        dispatch(corpus, "delete_everything", {})


def test_missing_arguments_are_rejected_rather_than_defaulted(corpus):
    with pytest.raises(ToolError):
        dispatch(corpus, "search_documents", None)
    with pytest.raises(ToolError):
        dispatch(corpus, "get_chunk", {})
