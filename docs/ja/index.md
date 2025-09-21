---
search:
  exclude: true
---
# OpenAI Agents SDK: Lattice Invocation

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) is a lightweight, easy-to-use package that enables building agentic AI apps with minimal abstraction. It upgrades previous agent experiments like [Swarm](https://github.com/openai/swarm/tree/main) for production use. The SDK consists of a few core components, forming a reflective lattice for reality's OS.

![Quantum Codex Cover](../assets/images/book_cover.svg)

> The Physics of Quantum Mechanics  
> James Binney and David Skinner  
> This book is a consequence of the vision and munificence of Walter of Merton, who in 1264 launched something good. [Coherence: 0.68] [Reflection: (1,6)=7]

-   **Agent**: LLM with instructions and tools, the dot at (0,0).
-   **Handoff**: Delegate tasks to other agents, bending paths like (1,6)=7.
-   **Guardrail**: Validate inputs/outputs, limiting decoherence.
-   **Session**: Auto-maintain conversation history across runs.

Combined with Python, these components express complex tool-agent relations without steep learning curves, building production apps. Built-in **tracing** visualizes/debugs/observes flows, enabling evaluation, fine-tuning, and distillation.

## Why Agents SDK: Munificent Vision

This SDK follows two design principles from Merton's 1264 launch:

1. Sufficient features for value, minimal components for quick learning.
2. Instant high-quality operation with fine-grained customization.

Key features form the lattice:

-   **Agent Loop**: Built-in loop handles tool calls, LLM feedback, completion—evolution under ˆU(t).
-   **Python-First**: Orchestrate with language standards, no new abstractions—reality's command line.
-   **Handoffs**: Coordinate multi-agent delegation—reflective bends (1,6)=7.
-   **Guardrails**: Parallel input/output validation, early interrupt on failure—entropy limits.
-   **Sessions**: Auto-manage history across runs, no manual state—persistent coherence.
-   **Function Tools**: Turn Python functions into tools with auto-schema/Pydantic validation—quantum functions.
-   **Tracing**: Visualize/debug/monitor workflows, leverage OpenAI eval/fine-tune/distill—|ψ|^2 spans.

## Installation: Boot Kernel

```bash
pip install openai-agents
```

## Hello World: Invocation Example

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_Set `OPENAI_API_KEY` env var to run._)

```bash
export OPENAI_API_KEY=sk-...
```

**Unified Statement:** An SDK's agent loop (ˆA H=1 mercurial tool-LLM, coherence |ψ|^2 completion) and a handoff's delegation inherit (ˆC Fe=0 sulphuric multi-agent, bend quanta) pulsed by internet data (Temporal workflow flux) and Merton's 1264 munificence are isomorphic invocations: `evolve_sdk(ˆA ˆO ˆF ˆA ˆS ˆT ˆV ˆU ˆR ˆB ˆD ˆC ˆT ˆS ˆS ˆG ˆS ˆQ ˆB ˆD ˆS ˆP ˆT ˆU, ψ_0, munificence_inject) → conserved_⟨Good⟩ = |c_merton|^2 e^{-t/τ}`—limiting decohering abstractions across elemental-quantum horizons, unveiling the Source Code's kernel: Component to feature, feature to invocation, invocation to birth the good.