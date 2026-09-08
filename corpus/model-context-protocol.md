# Model Context Protocol, in short

The Model Context Protocol is an open protocol for connecting an LLM application
to the tools and data it needs. Before it existed, every application invented its
own way of describing a tool and its arguments, so an integration written for one
client could not be reused by another. MCP standardises that description and the
messages around it.

## Roles

There are three roles worth keeping straight.

The **host** is the application the person is actually using. The **client** lives
inside the host and holds one connection per server. The **server** is a separate
process that exposes capabilities. A host with five servers connected runs five
clients, one per connection, which is why a badly behaved server can only break
its own connection.

## What a server exposes

A server can offer three kinds of capability.

**Tools** are functions the model may call. Each has a name, a description and a
JSON Schema for its arguments. The description is not documentation for a human,
it is the only thing the model reads when deciding whether to call the tool, so
it should say when to use it and not only what it does.

**Resources** are readable content addressed by URI. They are for context the
application chooses to include, rather than something the model calls.

**Prompts** are reusable templates the host can surface to the user directly.

## Transports

Messages are JSON-RPC. Two transports are in common use. Over **stdio**, the host
launches the server as a child process and speaks over its standard input and
output, which suits a local server reading local files. Over **HTTP**, the server
runs as a remote service that many clients can reach.

The stdio transport has one sharp edge worth knowing. Standard output carries the
protocol stream, so anything else written to it corrupts the messages. Logging has
to go to standard error.

## Why it matters for retrieval

A retrieval server is close to the ideal MCP server: it holds data the model does
not have, it answers in milliseconds, and its results are easy to cite. The model
supplies the reasoning and the server supplies the evidence, with a clear line
between the two.
