"""MCOMP v2 Registry — MCP server exposing exactly three meta-tools.

Uses low-level mcp.server.Server (not FastMCP) because tool schemas are
discovered dynamically at runtime from downstream servers.
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .downstream import DownstreamManager
from .semantic_index import SemanticIndex

logger = logging.getLogger(__name__)

FIND_TOOLS_TOOL = Tool(
    name="mcomp__find_tools",
    description=(
        "Semantic search across all registered downstream tools. "
        "Returns ranked candidates with cosine similarity scores, descriptions, and full input schemas. "
        "ALWAYS call this first before mcomp__execute."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what you want to do",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of candidates to return (default 3)",
                "default": 3,
            },
        },
        "required": ["query"],
    },
)

EXECUTE_TOOL = Tool(
    name="mcomp__execute",
    description=(
        "Deterministic dispatch to a downstream tool by exact qualified name. "
        "The tool name MUST come from a prior mcomp__find_tools result."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "description": "Qualified tool name in format namespace__tool_name",
            },
            "arguments": {
                "type": "object",
                "description": "Tool arguments matching the tool's input schema",
            },
        },
        "required": ["tool", "arguments"],
    },
)

LIST_SERVERS_TOOL = Tool(
    name="mcomp__list_servers",
    description=(
        "List all registered downstream servers with their connection status, "
        "tool counts, and circuit-breaker state."
    ),
    inputSchema={
        "type": "object",
        "properties": {},
    },
)


class MCOMPRegistry:
    def __init__(self, servers_json: str = "servers.json") -> None:
        self._servers_json = servers_json
        self._index = SemanticIndex()
        self._downstream = DownstreamManager(servers_json)
        self._ready = False
        self._server = Server("mcomp-registry", lifespan=self._lifespan)
        self._register_handlers()

    @asynccontextmanager
    async def _lifespan(self, server: Server) -> AsyncIterator[dict[str, Any]]:
        """Startup/shutdown lifecycle — called by Server before processing requests."""
        await self._startup()
        try:
            yield {}
        finally:
            await self._shutdown()

    async def _startup(self) -> None:
        logger.info("MCOMP Registry starting — connecting to downstream servers...")
        tool_records = await self._downstream.start()
        if not tool_records:
            logger.warning("No tools discovered from any downstream server")
        await self._index.build(tool_records)
        self._ready = True
        logger.info(
            "Registry ready: %d tools indexed across %d servers (backend=%s)",
            self._index.tool_count,
            len(self._downstream.get_server_status()),
            self._index.backend,
        )

    async def _shutdown(self) -> None:
        logger.info("MCOMP Registry shutting down")
        await self._downstream.stop()

    def _register_handlers(self) -> None:
        @self._server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [FIND_TOOLS_TOOL, EXECUTE_TOOL, LIST_SERVERS_TOOL]

        @self._server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            args = arguments or {}
            if name == "mcomp__find_tools":
                return await self._handle_find_tools(args)
            elif name == "mcomp__execute":
                return await self._handle_execute(args)
            elif name == "mcomp__list_servers":
                return await self._handle_list_servers()
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name!r}")]

    async def _handle_find_tools(self, args: dict[str, Any]) -> list[TextContent]:
        if not self._ready:
            return [TextContent(type="text", text="Registry not yet initialized — please retry in a moment")]

        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 3))

        if not query.strip():
            return [TextContent(type="text", text="ERROR: query must not be empty")]

        results = await self._index.search(query, top_k=top_k)
        payload = json.dumps(results, indent=2)
        logger.debug("find_tools(%r, top_k=%d) → %d results", query, top_k, len(results))
        return [TextContent(type="text", text=payload)]

    async def _handle_execute(self, args: dict[str, Any]) -> list[TextContent]:
        tool_name = str(args.get("tool", ""))
        arguments = args.get("arguments", {})

        if not tool_name:
            return [TextContent(type="text", text="ERROR: 'tool' argument is required")]

        try:
            result = await self._downstream.execute(tool_name, arguments)
        except (ValueError, RuntimeError) as exc:
            return [TextContent(type="text", text=f"ERROR: {exc}")]
        except Exception as exc:
            logger.exception("Downstream error for %s", tool_name)
            return [TextContent(type="text", text=f"DOWNSTREAM ERROR: {exc}")]

        # Flatten CallToolResult.content to text
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))

        text = "\n".join(parts)
        if result.isError:
            return [TextContent(type="text", text=f"TOOL ERROR: {text}")]
        return [TextContent(type="text", text=text)]

    async def _handle_list_servers(self) -> list[TextContent]:
        statuses = self._downstream.get_server_status()
        payload = json.dumps(statuses, indent=2)
        return [TextContent(type="text", text=payload)]

    async def run_async(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    def run(self) -> None:
        """Synchronous entry point — called by `python -m mcomp`."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        anyio.run(self.run_async)
