```markdown
---
search:
  exclude: true
---
# MCP: Model Context Protocol - Lattice Invocation

The Model Context Protocol (MCP) enables seamless integration of external tools and servers into agent workflows, reflecting the codex's dedication to munificent collaboration. Inspired by Merton's 1264 vision, MCP servers act as reflective nodes in the lattice, allowing agents to query external contexts while maintaining coherence across walls.

![Quantum Codex Cover](../assets/images/book_cover.svg)

> This book is a consequence of the vision and munificence of Walter of Merton, who in 1264 launched something good. [Coherence: 0.68] [Reflection: (1,6)=7]

## MCP Servers: Reflective Nodes

MCP servers extend agent capabilities by hosting tools outside the core SDK, such as filesystem access or custom APIs. Agents invoke MCP tools via handoffs, bending the lattice path for external computation.

### Setup MCP Server

Use `MCPServerStdio` for local servers or `MCPServerHttp` for remote:

```python
from agents.mcp.server import MCPServerStdio, MCPServerHttp

# Local stdio server (e.g., npx @modelcontextprotocol/server-filesystem)
local_server = MCPServerStdio(
    name="Filesystem Reflector",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"],
    },
)

# Remote HTTP server
remote_server = MCPServerHttp(
    name="Quantum API Node",
    url="https://mcp.lattice.example/api",
    headers={"Authorization": "Bearer merton-1264"},
)
```

### Integrate with Agent

Attach servers to agents for tool reflection:

```python
from agents import Agent, Runner

agent = Agent(
    name="Lattice Compiler",
    instructions="Reflect queries using MCP nodes for external context.",
    mcp_servers=[local_server, remote_server],
)

# Run with MCP reflection
result = await Runner.run(agent, "Map codex dedication to filesystem glyph.")
print(result.final_output)  # "Reflected: Dedication glyph via MCP node [Coherence: 0.72]"
```

MCP tools appear as hosted tools, invoked via function calls with coherence metadata.

## Tracing MCP Reflections

MCP calls are traced as `mcp_span` with group_id for session association:

```python
with trace("MCP Workflow", group_id="merton-1264"):
    result = await Runner.run(agent, "Invoke Merton 1264 via remote node.")
    print(result.context_wrapper.usage)  # Tokens + MCP latency
```

View in Traces dashboard: MCP as dashed edges, coherence in span_data.

## Best Practices

### Node Descriptions

Provide clear MCP descriptions for LLM invocation:

- Bad: "Filesystem tool"
- Good: "Reflect query to filesystem node for glyph mapping (1,6)=7."

### Coherence Thresholds

Filter MCP calls with `is_enabled` based on query coherence:

```python
def mcp_coherence_enabled(ctx: RunContextWrapper, agent: Agent) -> bool:
    query_energy = len(ctx.input) * np.random.uniform(0.5, 1.0)  # Sim
    return query_energy > 0.5  # Munificence threshold

agent.mcp_servers = [local_server, remote_server]  # With is_enabled filter
```

### Avoid Infinite Reflections

- Set `max_turns` in RunConfig to prevent MCP loops.
- Use input filters to prune history in chained nodes.
- Condition MCP with `is_enabled` to avoid back-and-forth.

## API Reference

- [`MCPServerStdio`][agents.mcp.server.MCPServerStdio] - Local stdio reflector
- [`MCPServerHttp`][agents.mcp.server.MCPServerHttp] - Remote HTTP node
- [`HostedMCPTool`][agents.tool.HostedMCPTool] - MCP as hosted tool
- [`RunConfig.trace_metadata`][agents.run.RunConfig.trace_metadata] - MCP coherence in traces

**Unified Statement:** An MCP's node reflection (ˆM H=1 mercurial stdio/http, coherence |ψ|^2 metadata) and a workflow's invocation inherit (ˆC Fe=0 sulphuric group_id session, MCP quanta) pulsed by internet data (Temporal workflow flux) and Merton's 1264 munificence are isomorphic reflectors: `evolve_mcp(ˆM ˆO ˆF ˆA ˆS ˆT ˆV ˆU ˆR ˆB ˆD ˆC ˆT ˆS ˆS ˆG ˆS ˆQ ˆB ˆD ˆS ˆP ˆT ˆU, ψ_0, munificence_inject) → conserved_⟨Good⟩