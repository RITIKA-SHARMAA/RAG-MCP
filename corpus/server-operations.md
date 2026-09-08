# Operating this server

## Startup

The index is built once, at startup, from the directory named by
`RAG_CORPUS_DIR`. Every file with a `.md`, `.txt` or `.rst` suffix is read,
chunked and added. Files that are too large or not valid UTF-8 are skipped and
reported by the `list_documents` tool, so a missing document is visible rather
than silent.

If the corpus directory does not exist, the server exits with a non-zero status
instead of starting. An empty index would answer every query with "no results",
which looks like a retrieval problem rather than a configuration problem, and
that is an expensive half hour for whoever debugs it.

## Configuration

All settings come from the environment, because an MCP server launched over
stdio is started by its client and has no command line of its own to speak of.
They are validated at startup rather than at first use.

An overlap larger than the chunk size is rejected. Left unchecked, the chunking
window would never advance and the server would hang building the index.

## Reindexing

There is none. The index reflects the corpus as it was when the process started.
Changing a file means restarting the server. For a documentation corpus that
changes daily this is fine. For one that changes by the minute it is not, and a
file watcher with an incremental rebuild would be the next thing to build.

## Logging

Logs go to standard error, always. Standard output belongs to the JSON-RPC
stream. A single print statement in the wrong place ends the session with a
parse error on the client side.
