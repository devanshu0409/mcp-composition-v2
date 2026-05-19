"""Entry point: python -m src  →  run the MCOMP registry as an MCP server over stdio."""
import sys
import os

# Ensure servers.json is resolved relative to the CWD of whoever launched us,
# not the package directory.
servers_json = os.environ.get("MCOMP_SERVERS", "servers.json")

from src.registry import MCOMPRegistry  # noqa: E402

MCOMPRegistry(servers_json=servers_json).run()
