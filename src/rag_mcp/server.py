"""MCP server wiring: build the corpus, register handlers, serve over stdio.

Nothing in this file makes a retrieval decision. It exists to connect the
handlers in ``tools.py`` to an MCP session, so the interesting code stays
testable without a transport.

Written against the 2.x Python SDK, where the low level ``Server`` takes its
handlers as constructor callbacks rather than decorators, and a handler returns
a full result object rather than a bare list of content.
"""

from __future__ import annotations

import logging
import sys

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .config import Settings
from .corpus import Corpus, CorpusError
from .tools import ToolError, dispatch, tool_definitions

SERVER_NAME = "rag-mcp"
SERVER_VERSION = "0.1.0"

# stdout carries the JSON-RPC stream, so every log line has to go to stderr.
# Writing a single stray line to stdout corrupts the protocol and the client
# disconnects with a parse error that points nowhere near the cause.
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVER_NAME)


def make_handlers(corpus: Corpus):
    """Build the two request handlers bound to one corpus.

    Returned rather than registered so the tests can call them directly. Neither
    reads the request context, so a test can pass ``None`` for it.
    """

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_definitions(corpus.settings.default_top_k))

    async def on_call_tool(
        ctx, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            content = dispatch(corpus, params.name, params.arguments)
        except ToolError as exc:
            # Returned as a tool error, not raised as a transport failure. The
            # model reads the message and can retry with corrected arguments,
            # which is the whole reason these messages are written for a reader.
            logger.warning("tool %s rejected: %s", params.name, exc)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))], is_error=True
            )
        return types.CallToolResult(content=content)

    return on_list_tools, on_call_tool


def build_server(corpus: Corpus) -> Server:
    on_list_tools, on_call_tool = make_handlers(corpus)
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "Search a local document corpus. Call search_documents first, then "
            "get_chunk or get_document to read the full text of anything you "
            "intend to quote."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    corpus = Corpus.build(settings)
    logger.info(
        "indexed %d documents into %d chunks from %s",
        len(corpus.documents),
        corpus.chunk_count,
        settings.corpus_dir,
    )
    for skipped in corpus.skipped:
        logger.warning("skipped %s", skipped)

    server = build_server(corpus)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:
    import anyio

    try:
        anyio.run(serve)
    except CorpusError as exc:
        # Fail loudly on startup rather than serving an empty index that
        # silently answers every query with "no results".
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("configuration error: %s", exc)
        return 1
    return 0
