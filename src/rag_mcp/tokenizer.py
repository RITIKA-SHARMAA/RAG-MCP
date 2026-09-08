"""Tokenisation used by both the indexer and the query path.

Both sides must agree exactly. If a query token is normalised differently from
the document token it should match, the term simply never scores, and the bug is
invisible: results come back, they are just quietly worse. Keeping one function
here is the cheapest guard against that.
"""

from __future__ import annotations

import re

# Split on anything that is not a letter, digit or underscore. Hyphenated words
# therefore become two tokens ("read-through" -> "read", "through"), which is
# what we want for recall: a query for "read through cache" still matches.
_SPLIT = re.compile(r"[^0-9a-z_]+")

# A deliberately small stop list. Aggressive stopword removal hurts short
# technical queries more than it helps, because words like "not" and "in" carry
# meaning in phrases such as "not in index".
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how in is it its of on or that
    the this to was were what when where which who will with
    """.split()
)

MIN_TOKEN_LENGTH = 2

# Endings that are not plurals, so stripping the final "s" would be wrong.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "os")


def _singularise(token: str) -> str:
    """Strip one trailing plural "s".

    Without it, a search for "embeddings" scores nothing against a passage that
    only says "embedding", which is a surprising miss for anyone using the
    server. This is not a real stemmer: it will not connect "retrieval" to
    "retrieve". It is one rule, applied identically to documents and queries, so
    even where it is linguistically wrong ("bus" would become "bu") both sides
    are wrong in the same way and matching still works.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith(_NOT_PLURAL_ENDINGS):
        return token[:-1]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-word characters, drop short tokens, singularise.

    Order is preserved. Duplicates are preserved too, because BM25 needs raw
    term frequencies.
    """
    tokens = [t for t in _SPLIT.split(text.lower()) if len(t) >= MIN_TOKEN_LENGTH]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_singularise(t) for t in tokens]
