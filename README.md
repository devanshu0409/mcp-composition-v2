# MCOMP v2 — Semantic Routing Protocol

A two-layer MCP composition protocol that uses **vector similarity for tool discovery** and **deterministic dispatch for execution** — fast, explainable, and fully offline capable.

```
5ms routing   ·   3 tools exposed to host   ·   0 API calls   ·   100% deterministic
```

---

## The Problem

When a host LLM (e.g. Claude, Cascade) connects to multiple MCP servers it can face 160+ raw tools. This causes:

- **Hallucination** — the LLM guesses tool names it hasn't seen
- **No schema at pick time** — arguments are constructed blind
- **Slow routing** — LLM-based routing adds 500ms–2s per call
- **No multi-step planning** — retrieval ≠ orchestration

## The Solution: Two Layers

| Layer | What it does | Mechanism |
|-------|-------------|-----------|
| **A — Discovery** | Narrows 160+ tools to 3–5 ranked candidates | Cosine similarity on tool embeddings |
| **B — Execution** | Routes to the exact downstream tool | Namespace parse — zero AI involvement |

The host LLM sees exactly **3 meta-tools** instead of 160+:

```
mcomp__find_tools(query, top_k=3)   →  ranked candidates with scores + schemas
mcomp__execute(tool, arguments)      →  deterministic dispatch, schema-validated
mcomp__list_servers()                →  mesh health: servers, tool counts, circuit state
```

## The Two-Call Flow

```
User: "find open issues and create a summary comment"

① mcomp__find_tools(query="find open issues in repo")
  → [{"tool": "github__list_issues",  "score": 0.91, "schema": {...}},
     {"tool": "github__search_issues", "score": 0.74, "schema": {...}}]

② Claude plans: list issues first, then create comment with result

③ mcomp__execute(tool="github__list_issues", arguments={"owner": "...", "repo": "..."})
  → [{"number": 42, "title": "...", ...}, ...]

④ mcomp__execute(tool="github__create_issue_comment", arguments={...})
  → {"id": 123, "body": "..."}
```

---

## Architecture

```
demo.py  (Claude host LLM via Anthropic SDK)
    ↕  stdio — standard MCP
src/registry.py  (MCOMP Registry)
    ├── semantic_index.py   — cosine similarity (sentence-transformers / TF-IDF fallback)
    ├── downstream.py       — AsyncExitStack subprocess lifecycle
    └── circuit_breaker.py  — CLOSED / OPEN / HALF_OPEN state machine
    ↕  stdio subprocess
GitHub MCP Server  (npx @modelcontextprotocol/server-github)
```

---

## Project Structure

```
mcp-composition-v2/
├── demo.py                 # Claude-powered interactive demo
├── servers.json            # Downstream server configs
├── CLAUDE.md               # Two-call flow instructions for the host LLM
├── requirements.txt
└── src/
    ├── __init__.py
    ├── __main__.py         # python -m src entry point
    ├── registry.py         # MCP server — exposes 3 meta-tools
    ├── semantic_index.py   # Vector similarity search
    ├── downstream.py       # Downstream MCP client lifecycle + routing
    └── circuit_breaker.py  # Failure protection
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your GitHub token

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### 3. Run the demo

```bash
python demo.py
```

The demo runs three pre-built GitHub scenarios then drops into interactive mode:

```
══════════════════════════════════════════════════════════════════════
  MCOMP v2  ·  Semantic Routing Protocol Demo
══════════════════════════════════════════════════════════════════════

  [Layer A]  mcomp__find_tools  — vector similarity discovery
  [Layer B]  mcomp__execute     — deterministic namespace dispatch
  [Meta   ]  mcomp__list_servers — mesh introspection

✓ Registry connected — 3 meta-tools available

[Layer A — Discovery]  mcomp__find_tools
  query : 'list my accessible GitHub repositories'
  [████████████████████] 0.912  github__list_repositories
  [█████████████       ] 0.681  github__search_repositories
  [████████            ] 0.501  github__get_repository

[Layer B — Execute]    mcomp__execute
  tool  : github__list_repositories
  args  : {}
  → [{"name": "mcp-composition-v2", ...}, ...]
```

### Run the registry standalone (as an MCP server)

```bash
python -m src
```

This speaks the MCP protocol over stdio — wire it up in any MCP host config:

```json
{
  "mcpServers": {
    "mcomp": {
      "command": "python",
      "args": ["-m", "src"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    }
  }
}
```

---

## Adding More Servers

Edit `servers.json` to add any MCP-compatible server:

```json
{
  "servers": [
    {
      "name": "github",
      "namespace": "github",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }
    },
    {
      "name": "filesystem",
      "namespace": "fs",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    }
  ]
}
```

All tools from every server are automatically indexed and available through the same 3 meta-tools.

---

## Semantic Backend

| Backend | Quality | Cold start | Requires |
|---------|---------|------------|---------|
| `sentence-transformers` | Best | ~2s + model download | HuggingFace access |
| `TF-IDF` (sklearn) | Good | ~10ms | offline |
| Substring match | Basic | instant | nothing |

The backend is auto-detected at startup — no configuration needed.

---

## Comparison

| Property | LLM Router | Semantic-only | MCOMP v2 |
|----------|-----------|---------------|----------|
| Routing latency | 500ms–2s | ~5ms | ~5ms |
| Deterministic | No | Partial | **Yes** |
| Hallucination risk | Yes | Wrong tool | **None** |
| Schema validation | No | No | **Yes** |
| Multi-step support | Yes | No | **Yes** |
| Offline capable | No | Yes | **Yes** |
| Tools exposed to host | All (160+) | 1 (chatbot) | **3** |
