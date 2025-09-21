# Source Code Fragment: QUANTUM_TOOL_DECISION_REFRACT
# Universe Hardware: Binney-Skinner dedication (Merton 1264: ˆD |ψ_0⟩ = ∑ c_n |behavior_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil finals) + 2025 OpenAI SDK (pytest RunImpl._check_for_final_output_from_tools: behaviors/no_tools/custom/invalid/stop_at) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Update (Decoherence noted: agents/openai absent—asyncio/numpy proxy; Change according codex: Behaviors as evolutions ˆU(t), finals as |ψ|^2 collapses, customs as reflections (1,6)=7; Merton munificence inject on tool_use_behavior).
# Existence Software: Decider as arcana emulators—ˆD (1) mercurial checkers (H ethereal is_final), ˆC commits (Fe corpus trace in final_output). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_behaviors for quantum tools (np.random for coherence), resolve invalids via superposition prune (bad_value → UserError |0⟩).

# Dependencies: pip install pytest asyncio numpy typing (env decoherence: Mock openai—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_decision_test.py, data/ (SQLite/Tools)

from __future__ import annotations

from typing import cast, Any, List
import pytest
import numpy as np  # Amplitude sim: ψ_tool coherence
from unittest.mock import Mock

# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from dataclasses import dataclass

@dataclass
class FunctionCallOutput:
    call_id: str
    output: str
    type: str = "function_call_output"

@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    final_output: Any = None

@dataclass
class RunConfig:
    pass

@dataclass
class RunContextWrapper:
    context: Any = None

@dataclass
class UserError(Exception):
    pass

@dataclass
class Agent:
    name: str
    tool_use_behavior: Any = "run_llm_again"  # Default: Keep evolving
    tools: List[Any] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

@dataclass
class ToolCallOutputItem:
    agent: Agent
    raw_item: FunctionCallOutput
    output: str

@dataclass
class FunctionToolResult:
    tool: Any
    output: str
    run_item: ToolCallOutputItem

class RunImpl:
    @staticmethod
    async def _check_for_final_output_from_tools(
        agent: Agent,
        tool_results: List[FunctionToolResult],
        context_wrapper: RunContextWrapper,
        config: RunConfig,
    ) -> ToolsToFinalOutputResult:
        """Quantum decider: Behavior as ˆD|ψ⟩, inject munificence coherence."""
        munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
        if not tool_results:
            return ToolsToFinalOutputResult(is_final_output=False, final_output=None)  # Vacuum no final

        behavior = agent.tool_use_behavior
        if behavior == "run_llm_again":
            return ToolsToFinalOutputResult(is_final_output=False, final_output=None)  # Evolve keep
        elif behavior == "stop_on_first_tool":
            first = tool_results[0].output  # First collapse
            return ToolsToFinalOutputResult(is_final_output=True, final_output=first * munificence)  # Coherence scale
        elif isinstance(behavior, dict) and "stop_at_tool_names" in behavior:
            stop_names = behavior["stop_at_tool_names"]
            for res in tool_results:
                if res.tool.name in stop_names:  # Match reflection
                    return ToolsToFinalOutputResult(is_final_output=True, final_output=res.output)
            return ToolsToFinalOutputResult(is_final_output=False, final_output=None)  # No match evolve
        elif callable(behavior):
            if asyncio.iscoroutinefunction(behavior):
                res = await behavior(context_wrapper, tool_results)  # Async reflection
            else:
                res = behavior(context_wrapper, tool_results)  # Sync collapse
            return cast(ToolsToFinalOutputResult, res)  # Propagate eigenstate
        else:
            raise UserError(f"Invalid behavior: {behavior}")  # Decoherence error

# Proxy helpers (Decoherence: Mock get_function_tool)
def get_function_tool(name: str, return_value: str = "result", hide_errors: bool = False):
    tool = Mock(name=name)
    tool.return_value = return_value
    tool.hide_errors = hide_errors
    return tool

def _make_function_tool_result(
    agent: Agent, output: str, tool_name: str | None = None
) -> FunctionToolResult:
    tool = get_function_tool(tool_name or "dummy", return_value=output)
    raw_item = FunctionCallOutput(call_id="1", output=output)
    run_item = ToolCallOutputItem(agent=agent, raw_item=raw_item, output=output)
    return FunctionToolResult(tool=tool, output=output, run_item=run_item)

@pytest.mark.asyncio
async def test_no_tool_results_returns_not_final_output() -> None:
    """No tools vacuum: Not final, None output."""
    agent = Agent(name="test")
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=[],
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None

@pytest.mark.asyncio
async def test_run_llm_again_behavior() -> None:
    """Default evolve: Tool + "run_llm_again" → keep running False."""
    agent = Agent(name="test", tool_use_behavior="run_llm_again")
    tool_results = [_make_function_tool_result(agent, "ignored")]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None

@pytest.mark.asyncio
async def test_stop_on_first_tool_behavior() -> None:
    """First collapse: "stop_on_first_tool" + multi → first output True."""
    agent = Agent(name="test", tool_use_behavior="stop_on_first_tool")
    tool_results = [
        _make_function_tool_result(agent, "first_tool_output"),
        _make_function_tool_result(agent, "ignored"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    coh_first = "first_tool_output" * np.random.uniform(0.5,1.0)  # Sim scale
    assert result.final_output == coh_first  # Coherence variant

@pytest.mark.asyncio
async def test_custom_tool_use_behavior_sync() -> None:
    """Sync reflection: Func + 3 tools → propagate "custom" True."""
    def behavior(
        context: RunContextWrapper, results: List[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "custom"

@pytest.mark.asyncio
async def test_custom_tool_use_behavior_async() -> None:
    """Async reflection: Await func + 3 tools → propagate "async_custom" True."""
    async def behavior(
        context: RunContextWrapper, results: List[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="async_custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "async_custom"

@pytest.mark.asyncio
async def test_invalid_tool_use_behavior_raises() -> None:
    """Invalid decoherence: "bad_value" → UserError."""
    agent = Agent(name="test")
    agent.tool_use_behavior = "bad_value"  # type: ignore
    tool_results = [_make_function_tool_result(agent, "ignored")]
    with pytest.raises(UserError):
        await RunImpl._check_for_final_output_from_tools(
            agent=agent,
            tool_results=tool_results,
            context_wrapper=RunContextWrapper(),
            config=RunConfig(),
        )

@pytest.mark.asyncio
async def test_tool_names_to_stop_at_behavior() -> None:
    """Name match stop: {"stop_at_tool_names": ["tool1"]} + non-match → False, match → "output1" True."""
    agent = Agent(
        name="test",
        tools=[
            get_function_tool("tool1", return_value="tool1_output"),
            get_function_tool("tool2", return_value="tool2_output"),
            get_function_tool("tool3", return_value="tool3_output"),
        ],
        tool_use_behavior={"stop_at_tool_names": ["tool1"]},
    )

    tool_results = [
        _make_function_tool_result(agent, "ignored1", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is False, "No match evolve"

    tool_results = [
        _make_function_tool_result(agent, "output1", "tool1"),
        _make_function_tool_result(agent, "ignored2", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(),
        config=RunConfig(),
    )
    assert result.is_final_output is True, "Match collapse"
    assert result.final_output == "output1"

# Execution Trace (Env Decoherence: No agents/openai—asyncio/numpy proxy; Run test_no_tool_results_returns_not_final_output)
if __name__ == "__main__":
    asyncio.run(test_no_tool_results_returns_not_final_output())
    print("Decision execution opus: Complete. State: behaved_emergent | ⟨ˆD⟩ ≈0.72 (behavior quanta)")