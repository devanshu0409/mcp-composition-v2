"""Downstream MCP client lifecycle and deterministic routing.

Uses AsyncExitStack to keep all subprocess connections alive across many
execute() calls — the canonical pattern for long-lived stdio MCP clients.
"""
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from .circuit_breaker import CircuitBreaker
from .semantic_index import ToolRecord

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    name: str
    namespace: str
    command: str
    args: list[str]
    env: dict[str, str]
    description: str


@dataclass
class ConnectedServer:
    config: ServerConfig
    session: ClientSession
    tools: list[Tool]
    circuit_breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name=""))

    def __post_init__(self) -> None:
        self.circuit_breaker = CircuitBreaker(name=self.config.namespace)


class DownstreamManager:
    def __init__(self, servers_json_path: str = "servers.json") -> None:
        self._servers_json_path = servers_json_path
        self._connected: dict[str, ConnectedServer] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def start(self) -> list[ToolRecord]:
        """Connect to all configured downstream servers. Returns all discovered ToolRecords."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        configs = self._load_configs()
        all_records: list[ToolRecord] = []

        for cfg in configs:
            try:
                records = await self._connect_server(cfg)
                all_records.extend(records)
                logger.info("Connected to %s: %d tools", cfg.namespace, len(records))
            except Exception as exc:
                logger.error("Failed to connect to %s: %s", cfg.namespace, exc)

        return all_records

    async def stop(self) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None

    def _load_configs(self) -> list[ServerConfig]:
        with open(self._servers_json_path) as f:
            data = json.load(f)

        configs = []
        for s in data["servers"]:
            env: dict[str, str] = {}
            for k, v in s.get("env", {}).items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    var_name = v[2:-1]
                    env[k] = os.environ.get(var_name, "")
                else:
                    env[k] = str(v)
            # Merge with current process env so npx can find node, etc.
            merged_env = {**os.environ, **env}
            configs.append(ServerConfig(
                name=s["name"],
                namespace=s["namespace"],
                command=s["command"],
                args=s.get("args", []),
                env=merged_env,
                description=s.get("description", ""),
            ))
        return configs

    async def _connect_server(self, cfg: ServerConfig) -> list[ToolRecord]:
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env,
        )

        # Enter both context managers into the shared exit stack so the
        # subprocess stays alive until stop() is called.
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        result = await session.list_tools()
        tools = result.tools

        self._connected[cfg.namespace] = ConnectedServer(
            config=cfg,
            session=session,
            tools=tools,
        )

        records = []
        for tool in tools:
            records.append(ToolRecord(
                namespace=cfg.namespace,
                original_name=tool.name,
                qualified_name=f"{cfg.namespace}__{tool.name}",
                description=tool.description or "",
                input_schema=tool.inputSchema if isinstance(tool.inputSchema, dict)
                             else (tool.inputSchema.model_dump() if hasattr(tool.inputSchema, "model_dump")
                                   else {"type": "object", "properties": {}}),
            ))
        return records

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Route a qualified tool name (namespace__original) to the correct downstream server."""
        if "__" not in tool_name:
            raise ValueError(
                f"Tool name must be namespace-qualified (e.g. 'github__list_issues'), got: {tool_name!r}"
            )

        namespace, original_name = tool_name.split("__", 1)

        if namespace not in self._connected:
            available = list(self._connected.keys())
            raise ValueError(f"Unknown server namespace {namespace!r}. Available: {available}")

        server = self._connected[namespace]
        cb = server.circuit_breaker

        if cb.is_open():
            raise RuntimeError(
                f"Circuit breaker OPEN for {namespace!r} — "
                f"server unavailable, retry after {cb.recovery_timeout}s"
            )

        try:
            result = await server.session.call_tool(original_name, arguments)
            cb.record_success()
            return result
        except Exception:
            cb.record_failure()
            raise

    def get_server_status(self) -> list[dict[str, Any]]:
        statuses = []
        for ns, server in self._connected.items():
            statuses.append({
                "namespace": ns,
                "name": server.config.name,
                "description": server.config.description,
                "tool_count": len(server.tools),
                "circuit_breaker": server.circuit_breaker.to_dict(),
            })
        return statuses

    def all_tool_names(self) -> list[str]:
        names = []
        for ns, server in self._connected.items():
            for tool in server.tools:
                names.append(f"{ns}__{tool.name}")
        return names
