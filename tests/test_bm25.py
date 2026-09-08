import pytest

from rag_mcp.bm25 import BM25Index


def build(*docs: str) -> BM25Index:
    index = BM25Index()
    for doc in docs:
        index.add(doc.split())
    return index


def test_a_rare_term_outscores_a_common_one():
    index = build(
        "cache cache lock",
        "cache miss",
        "cache write",
        "cache read",
    )
    # "lock" appears in one document of four, "cache" in all four, so a query
    # for "lock" should rank document 0 far above what "cache" alone earns it.
    assert index.idf("lock") > index.idf("cache")


def test_ranking_puts_the_denser_document_first():
    index = build("lock lock lock row", "lock row", "row column")
    ranked = index.search(["lock"], top_k=3)
    assert [i for i, _ in ranked] == [0, 1]


def test_documents_scoring_zero_are_not_returned():
    index = build("lock row", "cache miss")
    ranked = index.search(["queue"], top_k=5)
    assert ranked == []


def test_term_frequency_saturates():
    index = build("lock " * 2, "lock " * 20)
    two = index.score(["lock"], 0)
    twenty = index.score(["lock"], 1)
    # Ten times the occurrences must not give ten times the score.
    assert twenty > two
    assert twenty < two * 3


def test_length_normalisation_prefers_the_shorter_document():
    padding = " ".join(f"w{i}" for i in range(200))
    index = build("lock row", f"lock {padding}")
    assert index.score(["lock"], 0) > index.score(["lock"], 1)


def test_top_k_limits_results():
    index = build("lock a", "lock b", "lock c", "lock d")
    assert len(index.search(["lock"], top_k=2)) == 2


def test_ties_break_deterministically_by_index():
    index = build("lock row", "lock row", "lock row")
    first = index.search(["lock"], top_k=3)
    second = index.search(["lock"], top_k=3)
    assert first == second
    assert [i for i, _ in first] == [0, 1, 2]


def test_empty_query_returns_nothing_and_empty_index_is_safe():
    assert build("lock row").search([], top_k=3) == []
    assert BM25Index().search(["lock"], top_k=3) == []


def test_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be positive"):
        build("lock row").search(["lock"], top_k=0)


def test_idf_stays_non_negative_for_a_term_in_every_document():
    index = build("lock", "lock", "lock")
    assert index.idf("lock") >= 0
