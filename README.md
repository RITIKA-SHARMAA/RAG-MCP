# RAG-MCP

A Model Context Protocol server that gives an LLM client searchable access to a
local document corpus. It indexes a directory of Markdown and text files with
BM25 at startup and exposes four tools over stdio: search, read a passage, read a
document, list the corpus.

The retrieval is the whole product. Generation is the client's job, so this
server never calls a model and needs no API key.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest          # 67 tests
RAG_CORPUS_DIR=corpus .venv/bin/python -m rag_mcp
```

Run on its own it will sit waiting for JSON-RPC on stdin, which is correct: an
MCP server is launched by its client, not used directly.

### Connecting it to a client

Add it to the client's MCP server config. For Claude Desktop that is
`claude_desktop_config.json`; for Claude Code it is `.mcp.json` in the project.

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "rag_mcp"],
      "env": { "RAG_CORPUS_DIR": "/absolute/path/to/your/docs" }
    }
  }
}
```

Point `RAG_CORPUS_DIR` at any directory of notes. The `corpus/` directory in this
repo holds three documents about MCP and retrieval so the server does something
useful the first time it starts.

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `search_documents` | `query`, optional `top_k` (1 to 25) | Ranked passages with chunk id, source file, BM25 score and a snippet centred on the match |
| `get_chunk` | `chunk_id` | The full passage plus its character offsets in the source file |
| `get_document` | `source` | The whole document, truncated with a flag if very large |
| `list_documents` | none | Every indexed document with size and chunk count, plus anything that was skipped |

Responses are JSON in a text block. Two details are deliberate. Search returns
the `chunk_id` so the model can fetch the exact passage before quoting it, and an
empty result set comes back with an explicit note saying nothing matched, rather
than as an empty list a model might paper over.

## How retrieval works

**Tokenising.** Lowercase, split on non-word characters, drop tokens shorter than
two characters and a small stop list, then strip one trailing plural `s`.
Documents and queries go through the same function, which matters more than it
sounds: if the two sides normalise differently, terms silently stop matching and
results get quietly worse rather than visibly broken.

**Chunking.** Documents are split into overlapping windows of 180 words with 40
words of overlap, both configurable. Overlap exists so a fact sitting on a chunk
boundary still appears whole in one chunk. Each chunk keeps the character offsets
of its span, so any result can be traced back to an exact region of the source.

**Ranking.** BM25 Okapi, `k1 = 1.5`, `b = 0.75`:

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(D,Q) = sum over t in Q of
             idf(t) * f(t,D) * (k1 + 1) / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))
```

`k1` controls how fast term frequency saturates, so the tenth occurrence of a
word adds far less than the second. `b` controls length normalisation, so a long
document does not win simply by containing more words.

## Design decisions

**BM25 rather than embeddings.** No model download, no vector store, no API key,
so the server starts in the time it takes to walk a directory and runs anywhere
the client runs. Every score is explainable from the formula, so a surprising
ranking is debuggable. The cost is real: BM25 matches words, not meaning, and a
query phrased entirely in synonyms will miss. On technical documentation, where
the person asking usually shares the vocabulary of the text, that trade is worth
taking. It would not be for a corpus of customer emails.

**The index is in memory and built once.** For a documentation corpus this is a
few megabytes and it removes a whole class of failure: no stale index, no vector
store to be unreachable. The cost is that changing a file means restarting the
server.

**Failing loudly at startup.** A missing corpus directory exits non-zero instead
of serving an empty index. An empty index answers every query with "no results",
which reads as a retrieval bug and hides a configuration mistake.

**Logging to stderr, always.** Stdout carries the JSON-RPC stream. One stray
`print` corrupts the protocol and the client disconnects with a parse error that
points nowhere near the cause.

**Dispatch separated from transport.** `tools.py` takes a corpus, a tool name and
a dict, and returns content. It has no session object in it, so every behaviour
is unit-testable without starting a server, and `server.py` stays pure wiring.

## Tests

```bash
.venv/bin/python -m pytest -q
```

67 tests. They cover the tokeniser's normalisation rules, chunk overlap and
offset round-tripping, BM25 properties that are easy to get wrong (term frequency
saturation, length normalisation, non-negative idf, deterministic tie breaks),
corpus building including skipped and non-UTF-8 files, every tool's success and
failure path, the server handlers themselves, verifying that a bad tool call comes back as a
tool error the model can read rather than a crashed connection, and one end to
end test that runs a real client session against the server over in-memory
streams. That last one is the only test that catches a mistake in the wiring
itself, such as a handler that is never registered or a result shape the client
cannot parse.

Built against the 2.x Python SDK, where the low level `Server` takes its
handlers as constructor callbacks rather than decorators and `Tool` uses
`input_schema` rather than `inputSchema`. The dependency is pinned to `<3` so a
future major release cannot silently break it.

CI runs the suite on Python 3.11, 3.12 and 3.13 on every push.

## What is not built

Stated plainly, because a README that hides its gaps is worse than no README.

- **No semantic search.** Lexical matching only. See the trade above.
- **No stemming beyond plurals.** "retrieval" and "retrieve" are separate terms.
- **No incremental reindexing.** The index reflects the corpus at startup. A file
  watcher with an incremental rebuild is the obvious next step.
- **No hybrid ranking or reranking.** A second pass over the top results would
  likely improve precision more than any tuning of `k1` and `b`.
- **No PDF or HTML ingestion.** Markdown, text and reStructuredText only.
- **Whole-corpus scan per query.** Scoring iterates every chunk rather than using
  an inverted index with posting lists. Fine at this size, wrong at a large one.
