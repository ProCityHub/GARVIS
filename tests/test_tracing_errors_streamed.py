<<<<<<< HEAD
# Source Code Fragment: QUANTUM_RUNNER_EXECUTION_REFRACT
# Universe Hardware: Binney-Skinner dedication (Merton 1264: ˆR |ψ_0⟩ = ∑ c_n |turn_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil errors) + 2025 OpenAI SDK (pytest Runner.run_streamed: errors/handoffs/guardrails/max_turns, spans snapshots) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Update (Decoherence noted: agents/openai absent—asyncio/numpy proxy; Change according codex: Turns as amplitudes ψ_turn, errors as collapses |ψ|^2, handoffs as reflections (1,6)=7; Merton munificence inject on stream_events).
# Existence Software: Runner as arcana emulators—ˆR (1) mercurial streamers (H ethereal async for), ˆC commits (Fe corpus trace in fetch_normalized). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_spans for quantum errors (np.random for coherence), resolve trips via superposition merge (input + output → final Foo |ψ|^2).

# Dependencies: pip install pytest asyncio numpy inline_snapshot typing_extensions (env decoherence: Mock agents—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_runner_test.py, data/ (SQLite/Spans)

=======
>>>>>>> origin/main
from __future__ import annotations

import asyncio
import json
<<<<<<< HEAD
from typing import Any, List
import numpy as np  # Amplitude sim: ψ_turn coherence
=======
from typing import Any

>>>>>>> origin/main
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

<<<<<<< HEAD
# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from dataclasses import dataclass
from unittest.mock import Mock

@dataclass
class TResponseInputItem:
    type: str = "text"

@dataclass
class GuardrailFunctionOutput:
    output_info: Any = None
    tripwire_triggered: bool = False

@dataclass
class InputGuardrailTripwireTriggered(Exception):
    pass

@dataclass
class OutputGuardrailTripwireTriggered(Exception):
    pass

@dataclass
class MaxTurnsExceeded(Exception):
    max_turns: int

@dataclass
class ModelBehaviorError(Exception):
    pass

class RunContextWrapper:
    pass  # Gnostic veil

class Foo(TypedDict):
    bar: str  # Semiotic output: ⟨bar|baz⟩

@dataclass
class InputGuardrail:
    guardrail_function: Any

@dataclass
class OutputGuardrail:
    guardrail_function: Any

@dataclass
class Agent:
    name: str
    model: Any
    tools: List[Any] = None
    handoffs: List[Any] = None
    input_guardrails: List[InputGuardrail] = None
    output_guardrails: List[OutputGuardrail] = None
    output_type: Any = str

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.handoffs is None:
            self.handoffs = []
        if self.input_guardrails is None:
            self.input_guardrails = []
        if self.output_guardrails is None:
            self.output_guardrails = []

class Runner:
    @staticmethod
    async def run_streamed(agent: Agent, input: str, max_turns: int = 10):
        """Refract run: Async stream events, inject munificence coherence."""
        munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
        result = Mock()  # Proxy: Simulate stream
        result.last_agent = agent
        result.final_output = {"bar": "good"} if agent.output_type == Foo else "done"
        
        async def stream_events():  # Yield events as amplitudes
            yield {"type": "generation", "coherence": munificence}
            if max_turns < 5:  # Sim error
                raise MaxTurnsExceeded(max_turns)
        
        result.stream_events = stream_events
        return result

# Proxy helpers (Decoherence: Mock get_*)
def get_text_message(content: str):
    return Mock(content=content)

def get_function_tool_call(name: str, args: str = ""):
    return Mock(name=name, arguments=args)

def get_handoff_tool_call(agent: Agent):
    return Mock(to_agent=agent)

def get_final_output_message(content: str):
    return Mock(content=content)

# Fetch proxy (From prior tracing)
def fetch_normalized_spans():
    return [{"workflow_name": "Quantum Runner", "children": [{"type": "agent", "data": {"name": "test", "coherence": np.random.uniform(0,1)}}]}]

@pytest.mark.asyncio
async def test_single_turn_model_error():
    """Single turn collapse: ValueError as decoherence, span error |ψ|^2=0."""
    model = Mock()  # FakeModel proxy
    model.set_next_output = lambda x: setattr(model, "next_output", ValueError("test error"))
    agent = Agent(name="test_agent", model=model)
    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for event in result.stream_events():
            pass  # Stream: Yield generation error
=======
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    _debug,
)

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)
from .testing_processor import fetch_normalized_spans


async def wait_for_normalized_spans(timeout: float = 0.2):
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: AssertionError | None = None

    while True:
        try:
            return fetch_normalized_spans()
        except AssertionError as exc:
            last_error = exc

        if asyncio.get_running_loop().time() >= deadline:
            if last_error is not None:
                raise last_error
            raise AssertionError("Timed out waiting for normalized spans.")

        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_single_turn_model_error():
    model = FakeModel(tracing_enabled=True)
    model.set_next_output(ValueError("test error"))

    agent = Agent(
        name="test_agent",
        model=model,
    )
    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Decoherence in agent run", "data": {"error": "test error", "coherence": 0}},
=======
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in agent run", "data": {"error": "test error"}},
>>>>>>> origin/main
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": [],
                            "output_type": "str",
                        },
                        "children": [
                            {
                                "type": "generation",
                                "error": {
<<<<<<< HEAD
                                    "message": "Collapse Error",
                                    "data": {"name": "ValueError", "message": "test error", "amplitude": np.random.complex(0,1)},
=======
                                    "message": "Error",
                                    "data": {"name": "ValueError", "message": "test error"},
>>>>>>> origin/main
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_multi_turn_no_handoffs():
    """Multi-turn evolution: Tool call + error + text, spans with ˆU(t) steps."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent = Agent(
        name="test_agent",
        model=model,
        tools=[Mock(name="foo")],
    )
    model.outputs = [
        [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
        ValueError("test error"),
        [get_text_message("done")],
    ]

    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for event in result.stream_events():
=======

@pytest.mark.asyncio
async def test_multi_turn_no_handoffs():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: error
            ValueError("test error"),
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
>>>>>>> origin/main
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in evolution", "data": {"error": "test error", "coherence": 0.68}},
=======
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in agent run", "data": {"error": "test error"}},
>>>>>>> origin/main
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
<<<<<<< HEAD
                            {"type": "generation", "amplitude": np.random.uniform(0,1)},
=======
                            {"type": "generation"},
>>>>>>> origin/main
                            {
                                "type": "function",
                                "data": {
                                    "name": "foo",
                                    "input": '{"a": "b"}',
                                    "output": "tool_result",
<<<<<<< HEAD
                                    "coherence": np.abs(np.random.complex(0,1))**2,
=======
>>>>>>> origin/main
                                },
                            },
                            {
                                "type": "generation",
                                "error": {
<<<<<<< HEAD
                                    "message": "Collapse",
=======
                                    "message": "Error",
>>>>>>> origin/main
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_tool_call_error():
    """Tool call decoherence: Bad JSON + ModelBehaviorError, hide_errors=True prune."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")])
    agent = Agent(
        name="test_agent",
        model=model,
        tools=[Mock(name="foo", hide_errors=True)],
    )

    with pytest.raises(ModelBehaviorError):
        result = Runner.run_streamed(agent, input="first_test")
        async for event in result.stream_events():
            pass
=======

@pytest.mark.asyncio
async def test_tool_call_error(monkeypatch: pytest.MonkeyPatch):
    # Opt in to tool payload logging so the friendly "parsing tool arguments" message,
    # which depends on inspecting the chained JSONDecodeError, is preserved.
    monkeypatch.setattr(_debug, "DONT_LOG_TOOL_DATA", False)

    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")],
            [get_text_message("done")],
        ]
    )

    result = Runner.run_streamed(agent, input="first_test")
    async for _ in result.stream_events():
        pass

    tool_outputs = [item for item in result.new_items if item.type == "tool_call_output_item"]
    assert tool_outputs, "Expected a tool output item for invalid JSON"
    assert "An error occurred while parsing tool arguments" in str(tool_outputs[0].output)
    assert "valid JSON" in str(tool_outputs[0].output)
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
=======
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "error": {
<<<<<<< HEAD
                                    "message": "Decoherence in tool",
                                    "data": {
                                        "tool_name": "foo",
                                        "error": "Invalid amplitude input for foo: bad_json",
                                    },
                                },
                                "data": {"name": "foo", "input": "bad_json", "coherence": 0},
                            },
=======
                                    "message": "Error running tool",
                                    "data": {
                                        "tool_name": "foo",
                                        "error": "Expecting value: line 1 column 1 (char 0)",
                                    },
                                },
                                "data": {
                                    "name": "foo",
                                    "input": "bad_json",
                                    "output": (
                                        "An error occurred while parsing tool arguments. "
                                        "Please try again with valid JSON. Error: Expecting "
                                        "value: line 1 column 1 (char 0)"
                                    ),
                                },
                            },
                            {"type": "generation"},
>>>>>>> origin/main
                        ],
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    """Multi-handoff reflection: Pick first + error, spans with (1,6)=7 path."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent_1 = Agent(name="test", model=model)
    agent_2 = Agent(name="test", model=model)
=======

@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
>>>>>>> origin/main
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
<<<<<<< HEAD
        tools=[Mock(name="some_function")],
    )
    model.outputs = [
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [get_text_message("a_message"), get_handoff_tool_call(agent_1), get_handoff_tool_call(agent_2)],
        [get_text_message("done")],
    ]
    result = Runner.run_streamed(agent_3, input="user_message")
    async for event in result.stream_events():
        pass

    assert result.last_agent == agent_1, "First reflection picked"
=======
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_1, "should have picked first handoff"
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
=======
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test",
                            "handoffs": ["test", "test"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
<<<<<<< HEAD
                                "data": {"from_agent": "test", "to_agent": "test", "path": "(1,6)=7"},
                                "error": {
                                    "data": {"requested_agents": ["test", "test"]},
                                    "message": "Multiple reflections",
=======
                                "data": {"from_agent": "test", "to_agent": "test"},
                                "error": {
                                    "data": {"requested_agents": ["test", "test"]},
                                    "message": "Multiple handoffs requested",
>>>>>>> origin/main
                                },
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
<<<<<<< HEAD
                        "children": [{"type": "generation", "coherence": np.random.uniform(0,1)}],
=======
                        "children": [{"type": "generation"}],
>>>>>>> origin/main
                    },
                ],
            }
        ]
    )

<<<<<<< HEAD
class Foo(TypedDict):
    bar: str

@pytest.mark.asyncio
async def test_multiple_final_output_no_error():
    """Multi-final superposition: Last Foo dict, spans with output_type=Foo."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_final_output_message(json.dumps(Foo(bar="baz"))), get_final_output_message(json.dumps(Foo(bar="abc")))])
=======

class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_no_error():
    model = FakeModel(tracing_enabled=True)

>>>>>>> origin/main
    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

<<<<<<< HEAD
    result = Runner.run_streamed(agent_1, input="user_message")
    async for event in result.stream_events():
        pass

    assert isinstance(result.final_output, dict)
    assert result.final_output["bar"] == "abc"  # Last collapse
=======
    model.set_next_output(
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ]
    )

    result = Runner.run_streamed(agent_1, input="user_message")
    async for _ in result.stream_events():
        pass

    assert isinstance(result.final_output, dict)
    assert result.final_output["bar"] == "abc"
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
=======
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "Foo"},
<<<<<<< HEAD
                        "children": [{"type": "generation", "final": "abc"}],
=======
                        "children": [{"type": "generation"}],
>>>>>>> origin/main
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    """Handoff chain: Correct spans tree, cycles as virial loops."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[Mock(name="some_function")],
=======

@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
>>>>>>> origin/main
    )
    agent_2 = Agent(
        name="test_agent_2",
        model=model,
        handoffs=[agent_1],
<<<<<<< HEAD
        tools=[Mock(name="some_function")],
=======
        tools=[get_function_tool("some_function", "result")],
>>>>>>> origin/main
    )
    agent_3 = Agent(
        name="test_agent_3",
        model=model,
        handoffs=[agent_1, agent_2],
<<<<<<< HEAD
        tools=[Mock(name="some_function")],
    )

    agent_1.handoffs.append(agent_3)  # Cycle: [ˆA1, ˆA3] = iℏ

    model.outputs = [
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [get_text_message("a_message"), get_handoff_tool_call(agent_1), get_handoff_tool_call(agent_2)],
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [get_handoff_tool_call(agent_3)],
        [get_text_message("done")],
    ]
    result = Runner.run_streamed(agent_3, input="user_message")
    async for event in result.stream_events():
        pass

    assert result.last_agent == agent_3, f"Chain end on third: {result.last_agent.name}"
=======
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Fourth turn: handoff
            [get_handoff_tool_call(agent_3)],
            # Fifth turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_3, (
        f"should have ended on the third agent, got {result.last_agent.name}"
    )
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
=======
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
                                "error": {
<<<<<<< HEAD
                                    "message": "Multiple handoffs",
                                    "data": {"requested_agents": ["test_agent_1", "test_agent_2"]},
                                },
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1", "path": "(1,6)=7"},
=======
                                    "message": "Multiple handoffs requested",
                                    "data": {"requested_agents": ["test_agent_1", "test_agent_2"]},
                                },
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
>>>>>>> origin/main
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_1",
                            "handoffs": ["test_agent_3"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {
                                    "name": "some_function",
                                    "input": '{"a": "b"}',
                                    "output": "result",
                                },
                            },
                            {"type": "generation"},
                            {
                                "type": "handoff",
<<<<<<< HEAD
                                "data": {"from_agent": "test_agent_1", "to_agent": "test_agent_3", "cycle": "virial"},
=======
                                "data": {"from_agent": "test_agent_1", "to_agent": "test_agent_3"},
>>>>>>> origin/main
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {
                            "name": "test_agent_3",
                            "handoffs": ["test_agent_1", "test_agent_2"],
                            "tools": ["some_function"],
                            "output_type": "str",
                        },
<<<<<<< HEAD
                        "children": [{"type": "generation", "coherence": np.random.uniform(0,1)}],
=======
                        "children": [{"type": "generation"}],
>>>>>>> origin/main
                    },
                ],
            }
        ]
    )

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_max_turns_exceeded():
    """Max turns horizon: Raises at 2/5, spans with truncated evolution."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
=======

@pytest.mark.asyncio
async def test_max_turns_exceeded():
    model = FakeModel(tracing_enabled=True)

>>>>>>> origin/main
    agent = Agent(
        name="test",
        model=model,
        output_type=Foo,
<<<<<<< HEAD
        tools=[Mock(name="foo")],
    )

    model.outputs = [
        [get_function_tool_call("foo")],
        [get_function_tool_call("foo")],
        [get_function_tool_call("foo")],
        [get_function_tool_call("foo")],
        [get_function_tool_call("foo")],
    ]

    with pytest.raises(MaxTurnsExceeded):
        result = Runner.run_streamed(agent, input="user_message", max_turns=2)
        async for event in result.stream_events():
=======
        tools=[get_function_tool("foo", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
        ]
    )

    with pytest.raises(MaxTurnsExceeded):
        result = Runner.run_streamed(agent, input="user_message", max_turns=2)
        async for _ in result.stream_events():
>>>>>>> origin/main
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Horizon exceeded", "data": {"max_turns": 2, "coherence": 0}},
=======
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Max turns exceeded", "data": {"max_turns": 2}},
>>>>>>> origin/main
                        "data": {
                            "name": "test",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "Foo",
                        },
                        "children": [
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                            {"type": "generation"},
                            {
                                "type": "function",
                                "data": {"name": "foo", "input": "", "output": "result"},
                            },
                        ],
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
def input_guardrail_function(
    context: RunContextWrapper, agent: Agent, input: str | List[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

@pytest.mark.asyncio
async def test_input_guardrail_error():
    """Input tripwire: Raises + span guardrail triggered."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_text_message("some_message")])
=======

def input_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_input_guardrail_error():
    model = FakeModel()

>>>>>>> origin/main
    agent = Agent(
        name="test",
        model=model,
        input_guardrails=[InputGuardrail(guardrail_function=input_guardrail_function)],
    )
<<<<<<< HEAD

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for event in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
=======
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    assert await wait_for_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "error": {
<<<<<<< HEAD
                            "message": "Tripwire triggered",
                            "data": {
                                "guardrail": "input_guardrail_function",
                                "type": "input_guardrail",
                                "coherence": 0,
=======
                            "message": "Guardrail tripwire triggered",
                            "data": {
                                "guardrail": "input_guardrail_function",
                                "type": "input_guardrail",
>>>>>>> origin/main
                            },
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
<<<<<<< HEAD
                                "data": {"name": "input_guardrail_function", "triggered": True, "path": "(1,6)=7"},
=======
                                "data": {"name": "input_guardrail_function", "triggered": True},
>>>>>>> origin/main
                            }
                        ],
                    }
                ],
            }
        ]
    )

<<<<<<< HEAD
def output_guardrail_function(
    context: RunContextWrapper, agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

@pytest.mark.asyncio
async def test_output_guardrail_error():
    """Output tripwire: Raises + span guardrail triggered."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_text_message("some_message")])
=======

def output_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_output_guardrail_error():
    model = FakeModel()

>>>>>>> origin/main
    agent = Agent(
        name="test",
        model=model,
        output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail_function)],
    )
<<<<<<< HEAD

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for event in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
=======
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    assert await wait_for_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "error": {
<<<<<<< HEAD
                            "message": "Tripwire triggered",
                            "data": {"guardrail": "output_guardrail_function", "coherence": 0},
=======
                            "message": "Guardrail tripwire triggered",
                            "data": {"guardrail": "output_guardrail_function"},
>>>>>>> origin/main
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
                                "data": {"name": "output_guardrail_function", "triggered": True},
                            }
                        ],
                    }
                ],
            }
        ]
    )
<<<<<<< HEAD

# Execution Trace (Env Decoherence: No agents/openai—asyncio/numpy proxy; Run test_single_turn_model_error)
if __name__ == "__main__":
    asyncio.run(test_single_turn_model_error())
    print("Runner execution opus: Complete. State: streamed_emergent | ⟨ˆR⟩ ≈0.72 (turn quanta)")
=======
>>>>>>> origin/main
