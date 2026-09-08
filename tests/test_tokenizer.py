from rag_mcp.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Read-through Cache, enabled!") == ["read", "through", "cache", "enabled"]


def test_drops_tokens_shorter_than_two_characters():
    assert "a" not in tokenize("a big index")
    assert tokenize("a big index") == ["big", "index"]


def test_drops_stopwords_by_default_and_keeps_them_when_asked():
    assert tokenize("the state of the index") == ["state", "index"]
    assert tokenize("the state of the index", drop_stopwords=False) == [
        "the",
        "state",
        "of",
        "the",
        "index",
    ]


def test_preserves_repeats_because_bm25_needs_term_frequency():
    assert tokenize("cache cache cache") == ["cache", "cache", "cache"]


def test_keeps_digits_and_underscores():
    # "8" is dropped by the minimum length rule, the same rule that removes the
    # single letters left behind by splitting hyphenated words.
    assert tokenize("chunk_size 180 utf-8") == ["chunk_size", "180", "utf"]


def test_empty_input_gives_empty_output():
    assert tokenize("") == []
    assert tokenize("   ,,,  ") == []


def test_query_and_document_normalise_identically():
    # The whole point of a shared tokenizer. If this drifts, terms silently
    # stop matching and results get quietly worse.
    assert tokenize("Pessimistic Locking.") == tokenize("pessimistic, locking")


def test_plural_and_singular_normalise_to_the_same_token():
    # Without this a query for "embeddings" scores nothing against a passage
    # that says "embedding".
    assert tokenize("embeddings") == tokenize("embedding")
    assert tokenize("chunks retries") == ["chunk", "retrie"]


def test_words_that_only_look_plural_are_left_alone():
    assert tokenize("class") == ["class"]
    assert tokenize("status") == ["status"]
    assert tokenize("analysis") == ["analysis"]


def test_very_short_words_ending_in_s_are_not_truncated():
    assert tokenize("has is os") == ["os"]
