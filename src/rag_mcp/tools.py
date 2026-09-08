"""Tool schemas and dispatch.

The dispatch function is deliberately free of any MCP session or transport
object: it takes a corpus, a tool name and a dict of arguments, and returns
content. That keeps every behaviour in this file unit-testable without spawning
a server, and leaves ``server.py`` as pure wiring.

Every response is JSON in a text block. Models handle a compact JSON object more
reliably than prose, and it gives the client something it can parse if it wants
to render results itself.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent, Tool

from .corpus import Corpus
from .tokenizer import tokenize

SNIPPET_CHARS = 320
MAX_DOCUMENT_CHARS = 40_000


class ToolError(ValueError):
    """A tool was called with arguments the server cannot honour."""


def tool_definitions(default_top_k: int) -> list[Tool]:
    return [
        Tool(
            name="search_documents",
            description=(
                "Search the indexed corpus for passages relevant to a query and "
                "return ranked results with their source file and a snippet. Use "
                "this first, then call get_chunk or get_document to read the full "
                "text of anything you intend to quote or rely on."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or keyword query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"How many passages to return. Defaults to {default_top_k}.",
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_chunk",
            description=(
                "Return the full text of one passage by its chunk id, as returned "
                "by search_documents, together with its character offsets in the "
                "source file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Chunk id in the form 'path/to/file.md#3'.",
                    }
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="get_document",
            description=(
                "Return the full text of one indexed document by its source path. "
                "Large documents are truncated, and the response says so."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path relative to the corpus root.",
                    }
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="list_documents",
            description=(
                "List every indexed document with its size and chunk count. Use it "
                "to find out what this corpus actually covers before searching."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
    ]


def dispatch(corpus: Corpus, name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}
    if name == "search_documents":
        payload = _search(corpus, args)
    elif name == "get_chunk":
        payload = _get_chunk(corpus, args)
    elif name == "get_document":
        payload = _get_document(corpus, args)
    elif name == "list_documents":
        payload = _list_documents(corpus)
    else:
        raise ToolError(f"unknown tool: {name}")
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


def _search(corpus: Corpus, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")

    top_k = args.get("top_k", corpus.settings.default_top_k)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ToolError("top_k must be an integer")
    if not 1 <= top_k <= 25:
        raise ToolError("top_k must be between 1 and 25")

    hits = corpus.search(query, top_k)
    results = [
        {
            "chunk_id": hit.chunk.chunk_id,
            "source": hit.chunk.source,
            "score": hit.score,
            "snippet": _snippet(hit.chunk.text, query),
        }
        for hit in hits
    ]
    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        # An empty result set is a real answer, not a failure. Saying so plainly
        # stops a model inventing a passage to fill the gap.
        "note": None if results else "No indexed passage matched this query.",
    }


def _get_chunk(corpus: Corpus, args: dict[str, Any]) -> dict[str, Any]:
    chunk_id = args.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ToolError("chunk_id must be a non-empty string")
    try:
        chunk = corpus.get_chunk(chunk_id)
    except KeyError:
        raise ToolError(f"no such chunk: {chunk_id}") from None
    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "ordinal": chunk.ordinal,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "text": chunk.text,
    }


def _get_document(corpus: Corpus, args: dict[str, Any]) -> dict[str, Any]:
    source = args.get("source")
    if not isinstance(source, str) or not source:
        raise ToolError("source must be a non-empty string")
    try:
        document = corpus.get_document(source)
    except KeyError:
        raise ToolError(f"no such document: {source}") from None
    text = document.text
    truncated = len(text) > MAX_DOCUMENT_CHARS
    return {
        "source": document.source,
        "chunk_count": document.chunk_count,
        "truncated": truncated,
        "text": text[:MAX_DOCUMENT_CHARS] if truncated else text,
    }


def _list_documents(corpus: Corpus) -> dict[str, Any]:
    return {
        "document_count": len(corpus.documents),
        "chunk_count": corpus.chunk_count,
        "documents": [
            {
                "source": doc.source,
                "characters": len(doc.text),
                "chunks": doc.chunk_count,
            }
            for doc in sorted(corpus.documents.values(), key=lambda d: d.source)
        ],
        "skipped": corpus.skipped,
    }


def _snippet(text: str, query: str) -> str:
    """A window around the first query term that appears in the passage.

    Returning the head of the chunk regardless of where the match is makes
    results look wrong even when the ranking is right, so the window is centred
    on the match instead.
    """
    if len(text) <= SNIPPET_CHARS:
        return text
    lowered = text.lower()
    position = -1
    for token in tokenize(query):
        position = lowered.find(token)
        if position != -1:
            break
    if position == -1:
        return text[:SNIPPET_CHARS].rstrip() + " ..."
    start = max(0, position - SNIPPET_CHARS // 3)
    end = min(len(text), start + SNIPPET_CHARS)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
