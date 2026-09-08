# Retrieval design notes

## Why BM25 here

BM25 is a lexical ranking function. It scores a document against a query using
three signals: how often each query term appears in the document, how rare that
term is across the corpus, and how long the document is relative to the average.

It has two properties that matter for a small server. It needs no model, so
startup is a directory walk rather than a download, and every score can be
explained from the formula, so a surprising ranking is debuggable rather than
mysterious.

Its weakness is real and should be stated plainly. BM25 matches words, not
meaning. A query for "how do I stop the process" will not match a passage that
only ever says "terminating a worker". Embedding based retrieval handles that
paraphrase and BM25 does not. On technical documentation, where the person asking
usually already knows the vocabulary, the gap is smaller than it sounds.

## Term frequency saturation

The k1 parameter controls how quickly repeated terms stop adding score. Without
saturation, a document that repeats a word forty times would outrank a document
that uses it twice in a genuinely relevant sentence. With k1 around 1.5, the
second occurrence adds a lot, the tenth adds very little.

## Length normalisation

The b parameter controls how much a long document is penalised. At b equal to
zero, length is ignored, and long documents win simply by containing more words.
At b equal to one, length is fully normalised. The usual default of 0.75 sits
close to full normalisation while leaving some credit for a document that covers
a topic at length.

## Chunking

Documents are split into overlapping windows rather than indexed whole. Two
reasons. A whole document dilutes its own relevance signal, because a long file
about ten topics scores only moderately for each. And the passage handed back to
the model should be small enough to read, since an agent that receives a whole
file will spend its context on parts it did not need.

The overlap exists for boundaries. A fact that falls across a chunk edge would
otherwise be split in half and never rank for either fragment. Repeating the tail
of each chunk at the head of the next means it appears whole somewhere.

The cost of overlap is index size and duplicate results. A larger overlap means
more chunks and more near-identical passages competing for the same query.
