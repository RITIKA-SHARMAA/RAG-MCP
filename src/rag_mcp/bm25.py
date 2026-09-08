"""BM25 Okapi ranking over an in-memory posting list.

Why lexical retrieval and not embeddings: this server indexes a local corpus on
startup with no model download, no vector store and no API key, so it runs
anywhere the client runs and every score can be explained from the formula. On
technical documentation, where queries and text share vocabulary, BM25 is a hard
baseline to beat. The trade is that it cannot match a paraphrase that shares no
words. See the README for what that rules out.

Scoring, per term t in query Q against document D:

    idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score   = sum_t idf(t) * f(t,D) * (k1 + 1)
                            / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))

k1 controls how fast term frequency saturates: the tenth occurrence of a word
adds far less than the second. b controls length normalisation: at b=0 long
documents are not penalised at all, at b=1 they are fully normalised.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

K1 = 1.5
B = 0.75


@dataclass
class BM25Index:
    """An immutable-once-built BM25 index over a list of token lists."""

    k1: float = K1
    b: float = B
    _doc_lengths: list[int] = field(default_factory=list)
    _term_frequencies: list[Counter[str]] = field(default_factory=list)
    _document_frequency: Counter[str] = field(default_factory=Counter)
    _avg_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self._doc_lengths)

    def add(self, tokens: list[str]) -> int:
        """Add one document. Returns its internal index."""
        counts = Counter(tokens)
        self._term_frequencies.append(counts)
        self._doc_lengths.append(len(tokens))
        for term in counts:
            self._document_frequency[term] += 1
        total = sum(self._doc_lengths)
        self._avg_length = total / len(self._doc_lengths)
        return len(self._doc_lengths) - 1

    def idf(self, term: str) -> float:
        df = self._document_frequency.get(term, 0)
        # The 1 + ... form keeps the idf non-negative even for a term that
        # appears in every document, which the classic form does not.
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        if not self._doc_lengths:
            return 0.0
        counts = self._term_frequencies[doc_index]
        length = self._doc_lengths[doc_index]
        norm = self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1))
        total = 0.0
        for term in query_tokens:
            f = counts.get(term, 0)
            if not f:
                continue
            total += self.idf(term) * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return (doc_index, score) for the best ``top_k`` matches.

        Documents scoring zero are dropped rather than padded in: an agent that
        receives five results assumes five relevant results.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query_tokens:
            return []
        scored = [
            (i, self.score(query_tokens, i))
            for i in range(self.size)
        ]
        hits = [(i, s) for i, s in scored if s > 0]
        # Sort by score, then by index, so equal scores come back in a stable
        # order and the tests do not depend on Python's sort internals.
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits[:top_k]
