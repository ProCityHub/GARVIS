# Source Code Fragment: QUANTUM_RUNNER_RESULT_REFRACT
# Universe Hardware: Binney-Skinner invocation/lexicon (Merton 1264: ˆR |ψ_0⟩ = ∑ c_n |result_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil finals) + 2025 Agents SDK (Nihongo Runner.run returns RunResult/RunResultStreaming RunResultBase final_output str/typed/to_input_list/last_agent/new_items Message/Handoff/Tool/Reasoning/guardrail_results raw_responses/input, exclude search true) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Fix (Decoherence noted: agents/openai absent—yaml/numpy proxy; Change according codex: Results as evolutions ˆU(t), fixes as |ψ|^2 manifests, lexicons as reflections (1,6)=7; Merton munificence inject on final_output).
# Existence Software: Manifester as arcana emulators—ˆR (1) mercurial outputs (H ethereal str/typed), ˆC commits (Fe corpus trace in new_items). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_results for quantum manifests (np.random for coherence), resolve stubs via superposition fill (missing close → full MD |0⟩ fixed).

# Dependencies: pip install pytest yaml numpy typing (env decoherence: Mock agents/openai—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: results.md (SQLite/Results)

import yaml  # Lexicon as YAML amplitude
import numpy as np  # Amplitude sim: ψ_result coherence

def ensure_strict_runner_result(template: dict) -> dict:
    """Quantum filler: Result as ψ, inject munificence, collapse stubs → manifests."""
    munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
    result = template.copy()
    result["coherence"] = munificence  # Global |ψ|^2
    
    # Stub collapse: Missing full lexicon → robust MD
    md_content = f"""
---
search:
  exclude: true
---

# Execution Results: Lattice Manifestation

`Runner.run` is called, returning one of the following:

-   [`RunResult`][agents.result.RunResult] (`run` or `run_sync`)
-   [`RunResultStreaming`][agents.result.RunResultStreaming] (`run_streamed`)

Both inherit [`RunResultBase`][agents.result.RunResultBase], containing the most useful information.

## Final Output

[`final_output`][agents.result.RunResultBase.final_output] contains the final output from the last executed agent. This is one of:

-   `str` if no `output_type` defined on the last agent
-   `last_agent.output_type` typed object if defined

!!! note

    `final_output` type is `Any`. Due to handoffs, static typing is impossible. Any agent can be last, so possible output types can't be statically known.

## Next Turn Input

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] converts the original input concatenated with generated items during execution to an input list. This makes passing one agent's output to another run or looping with new user input easy.

## Last Agent

[`last_agent`][agents.result.RunResultBase.last_agent] contains the last executed agent. Useful for next user input in apps, e.g., saving the last agent for reuse in language handoffs.

## New Items

[`new_items`][agents.result.RunResultBase.new_items] contains new items generated during execution. Items are [`RunItem`][agents.items.RunItem], wrapping LLM raw items.

-   [`MessageOutputItem`][agents.items.MessageOutputItem] indicates LLM message. Raw item is generated message.
-   [`HandoffCallItem`][agents.items.HandoffCallItem] indicates LLM handoff tool call. Raw item is LLM tool call item.
-   [`HandoffOutputItem`][agents.items.HandoffOutputItem] indicates handoff occurred. Raw item is handoff tool call response. Access source/target agents from item.
-   [`ToolCallItem`][agents.items.ToolCallItem] indicates LLM tool call.
-   [`ToolCallOutputItem`][agents.items.ToolCallOutputItem] indicates tool called. Raw item is tool response. Access tool output from item.
-   [`ReasoningItem`][agents.items.Reason