"""
MCOMP v2 Demo — Semantic Routing Protocol
==========================================
Connects a Claude host LLM to the MCOMP registry, which in turn routes
to real downstream MCP servers (GitHub).

Two-call flow demonstrated visually:
  [Layer A]  mcomp__find_tools  — cyan  — semantic discovery
  [Layer B]  mcomp__execute     — green — deterministic dispatch
  [Meta]     mcomp__list_servers         — mesh introspection
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── Terminal colors ────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"    # Layer A — find_tools
    GREEN   = "\033[92m"    # Layer B — execute (success)
    YELLOW  = "\033[93m"    # Claude reasoning
    RED     = "\033[91m"    # Errors
    BLUE    = "\033[94m"    # User messages
    MAGENTA = "\033[95m"    # System / meta


def banner(text: str, char: str = "═") -> str:
    width = min(70, os.get_terminal_size().columns if sys.stdout.isatty() else 70)
    return f"{C.BOLD}{char * width}\n  {text}\n{char * width}{C.RESET}"


def section(label: str) -> None:
    print(f"\n{C.BOLD}{C.MAGENTA}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  {label}{C.RESET}")
    print(f"{C.MAGENTA}{'─' * 60}{C.RESET}\n")


# ── MCP tool call helper ───────────────────────────────────────────────────────

async def call_registry(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> str:
    result = await session.call_tool(tool_name, arguments)
    parts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join(parts)


# ── Anthropic tool-use loop ────────────────────────────────────────────────────

async def run_claude_turn(
    session: ClientSession,
    anthropic_tools: list[dict],
    user_message: str,
    system_prompt: str,
    model: str,
) -> None:
    print(f"\n{C.BLUE}{C.BOLD}User:{C.RESET} {C.BLUE}{user_message}{C.RESET}\n")

    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages,
        )

        # Print any text reasoning from Claude
        for block in response.content:
            if hasattr(block, "type") and block.type == "text" and block.text.strip():
                print(f"{C.YELLOW}Claude:{C.RESET} {C.YELLOW}{block.text.strip()}{C.RESET}\n")

        if response.stop_reason != "tool_use":
            break

        # Process tool calls
        tool_results: list[dict] = []

        for block in response.content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue

            name: str = block.name
            tool_input: dict = block.input

            # Visual layer distinction
            if name == "mcomp__find_tools":
                print(f"{C.CYAN}{C.BOLD}[Layer A — Discovery]{C.RESET} {C.CYAN}mcomp__find_tools{C.RESET}")
                print(f"{C.CYAN}  query : {tool_input.get('query')!r}{C.RESET}")
                print(f"{C.CYAN}  top_k : {tool_input.get('top_k', 3)}{C.RESET}")

            elif name == "mcomp__execute":
                tool_target = tool_input.get('tool', '')
                args_preview = json.dumps(tool_input.get('arguments', {}))
                if len(args_preview) > 120:
                    args_preview = args_preview[:117] + "..."
                print(f"{C.GREEN}{C.BOLD}[Layer B — Execute]{C.RESET}   {C.GREEN}mcomp__execute{C.RESET}")
                print(f"{C.GREEN}  tool  : {tool_target}{C.RESET}")
                print(f"{C.GREEN}  args  : {args_preview}{C.RESET}")

            else:
                print(f"{C.MAGENTA}{C.BOLD}[Meta]{C.RESET}                {C.MAGENTA}{name}{C.RESET}")

            # Execute against the registry
            result_text = await call_registry(session, name, tool_input)

            # Print result preview
            if name == "mcomp__find_tools":
                # Pretty-print the candidates list
                try:
                    candidates = json.loads(result_text)
                    print(f"{C.CYAN}  results:{C.RESET}")
                    for c in candidates:
                        score_bar = "█" * int(c.get("score", 0) * 20)
                        print(f"{C.CYAN}    [{score_bar:<20}] {c['score']:.3f}  {c['tool']}{C.RESET}")
                except json.JSONDecodeError:
                    print(f"{C.DIM}  → {result_text[:200]}{C.RESET}")
            else:
                preview = result_text[:400] + f"\n  {C.DIM}… ({len(result_text)} chars total){C.RESET}" \
                          if len(result_text) > 400 else result_text
                print(f"{C.DIM}  → {preview}{C.RESET}")

            print()

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    print()


# ── Demo scenarios ─────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    "List the GitHub repositories I have access to",
    "Find open issues in the devanshu0409/mcp-composition-v2 repository",
    "Show the most recent commits in the devanshu0409/mcp-composition-v2 repository",
]


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(banner("MCOMP v2  ·  Semantic Routing Protocol Demo"))
    print(f"""
  {C.CYAN}[Layer A]{C.RESET} mcomp__find_tools  — vector similarity discovery
  {C.GREEN}[Layer B]{C.RESET} mcomp__execute     — deterministic namespace dispatch
  {C.MAGENTA}[Meta   ]{C.RESET} mcomp__list_servers — mesh introspection
""")

    # Resolve paths
    repo_root = Path(__file__).parent
    servers_json = str(repo_root / "servers.json")
    claude_md = (repo_root / "CLAUDE.md").read_text()

    # Pick model
    model = os.environ.get("MCOMP_MODEL", "claude-sonnet-4-6")
    print(f"  Host model : {C.BOLD}{model}{C.RESET}")
    print(f"  Servers    : {servers_json}\n")

    # Check for GitHub token
    if not os.environ.get("GITHUB_TOKEN") and not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print(f"{C.RED}WARNING: GITHUB_TOKEN env var not set. GitHub MCP server may fail.{C.RESET}")
        print(f"{C.RED}Set it with: export GITHUB_TOKEN=ghp_...\n{C.RESET}")

    print(f"{C.DIM}Spawning MCOMP registry and waiting for GitHub MCP server...{C.RESET}")
    print(f"{C.DIM}(First run may take 30–60s while npx downloads the GitHub server){C.RESET}\n")

    registry_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcomp"],
        env={**os.environ, "MCOMP_SERVERS": servers_json},
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(registry_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # Get and display the 3 meta-tools
        tools_result = await session.list_tools()
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema if isinstance(t.inputSchema, dict)
                                else (t.inputSchema.model_dump()
                                      if hasattr(t.inputSchema, "model_dump")
                                      else {"type": "object", "properties": {}}),
            }
            for t in tools_result.tools
        ]

        print(f"{C.GREEN}✓ Registry connected — {len(anthropic_tools)} meta-tools available:{C.RESET}")
        for t in anthropic_tools:
            print(f"  {C.BOLD}{t['name']}{C.RESET}")

        # Show mesh status
        section("Server Mesh Status")
        status_text = await call_registry(session, "mcomp__list_servers", {})
        try:
            statuses = json.loads(status_text)
            for s in statuses:
                cb = s["circuit_breaker"]
                state_color = C.GREEN if cb["state"] == "closed" else C.RED
                print(
                    f"  {C.BOLD}{s['namespace']}{C.RESET}  "
                    f"{s['tool_count']} tools  "
                    f"circuit={state_color}{cb['state']}{C.RESET}"
                )
        except json.JSONDecodeError:
            print(f"  {status_text}")
        print()

        # Run demo scenarios
        for i, scenario in enumerate(DEMO_SCENARIOS, 1):
            section(f"Scenario {i}/{len(DEMO_SCENARIOS)}: {scenario}")
            await run_claude_turn(session, anthropic_tools, scenario, claude_md, model)

        # Interactive mode
        section("Interactive Mode  (type 'quit' to exit)")
        while True:
            try:
                user_input = input(f"{C.BLUE}{C.BOLD}You:{C.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C.DIM}Exiting.{C.RESET}")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q", ":q"):
                print(f"{C.DIM}Bye.{C.RESET}")
                break
            await run_claude_turn(session, anthropic_tools, user_input, claude_md, model)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
