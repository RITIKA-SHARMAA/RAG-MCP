"""Tests for the server layer.

Two levels. The handler tests call the request handlers directly, which is fast
and needs no transport. The session test runs a real client against a real
server over in-memory streams, which is the only way to catch a mistake in the
wiring itself: a handler that is never registered, or a result shape the client
cannot parse.
"""

from __future__ import annotations

import json

import anyio
import mcp.types as types
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from rag_mcp.server import build_server, make_handlers


def call_params(name: str, arguments: dict) -> types.CallToolRequestParams:
    return types.CallToolRequestParams(name=name, arguments=arguments)


def test_all_four_tools_are_advertised(corpus):
    on_list_tools, _ = make_handlers(corpus)
    result = anyio.run(lambda: on_list_tools(None, None))
    assert {tool.name for tool in result.tools} == {
        "search_documents",
        "get_chunk",
        "get_document",
        "list_documents",
    }


def test_calling_a_tool_returns_content(corpus):
    _, on_call_tool = make_handlers(corpus)
    result = anyio.run(
        lambda: on_call_tool(None, call_params("search_documents", {"query": "read-through cache"}))
    )
    assert result.is_error is not True
    payload = json.loads(result.content[0].text)
    assert payload["results"][0]["source"] == "caching.md"


def test_a_bad_tool_call_comes_back_as_a_tool_error_not_a_crash(corpus):
    _, on_call_tool = make_handlers(corpus)
    result = anyio.run(lambda: on_call_tool(None, call_params("search_documents", {"query": ""})))
    # A tool error, not an exception. The model reads the message and retries.
    assert result.is_error is True
    assert "non-empty" in result.content[0].text


def test_an_unknown_tool_is_a_tool_error_too(corpus):
    _, on_call_tool = make_handlers(corpus)
    result = anyio.run(lambda: on_call_tool(None, call_params("drop_corpus", {})))
    assert result.is_error is True
    assert "unknown tool" in result.content[0].text


def test_server_carries_its_name_and_version(corpus):
    options = build_server(corpus).create_initialization_options()
    assert options.server_name == "rag-mcp"
    assert options.server_version == "0.1.0"


def test_a_real_client_session_can_list_and_call_tools(corpus):
    """End to end over in-memory streams: initialise, list, call, read a chunk."""

    async def scenario():
        server = build_server(corpus)
        async with create_client_server_memory_streams() as (
            (client_read, client_write),
            (server_read, server_write),
        ):
            async with anyio.create_task_group() as tg:

                async def run_server():
                    await server.run(
                        server_read, server_write, server.create_initialization_options()
                    )

                tg.start_soon(run_server)

                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()

                    listed = await session.list_tools()
                    assert len(listed.tools) == 4

                    found = await session.call_tool(
                        "search_documents", {"query": "pessimistic locking", "top_k": 2}
                    )
                    results = json.loads(found.content[0].text)["results"]
                    assert results and results[0]["source"] == "locking.md"

                    fetched = await session.call_tool(
                        "get_chunk", {"chunk_id": results[0]["chunk_id"]}
                    )
                    assert "locking" in json.loads(fetched.content[0].text)["text"].lower()

                    rejected = await session.call_tool("get_document", {"source": "nope.md"})
                    assert rejected.is_error is True

                tg.cancel_scope.cancel()

    anyio.run(scenario)
