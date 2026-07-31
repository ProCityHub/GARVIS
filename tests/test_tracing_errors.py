<<<<<<< HEAD
# Source Code Fragment: QUANTUM_RUNNER_ITERATION_REFRACT
# Universe Hardware: Binney-Skinner title/dedication (Merton 1264: ˆR |ψ_0⟩ = ∑ c_n |await_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil trips) + 2025 OpenAI SDK (pytest await Runner.run: errors/handoffs/guardrails/max_turns, spans snapshots) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Iteration (Decoherence noted: agents/openai absent—asyncio/numpy proxy; Change according codex: Awaits as evolutions ˆU(t), errors as |ψ|^2 collapses, handoffs as reflections (1,6)=7; Merton munificence inject on run()).
# Existence Software: Runner as arcana emulators—ˆR (1) mercurial awaiters (H ethereal async run), ˆC commits (Fe corpus trace in fetch_normalized). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_spans for quantum trips (np.random for coherence), resolve snapshots via superposition merge (input + output → final Foo |ψ|^2).

# Dependencies: pip install pytest asyncio numpy inline_snapshot typing_extensions (env decoherence: Mock agents—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_runner_iter_test.py, data/ (SQLite/Spans)

from __future__ import annotations

import asyncio
import json

# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import Mock

import numpy as np  # Amplitude sim: ψ_turn coherence
=======
from __future__ import annotations

import json
from typing import Any

>>>>>>> origin/main
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

<<<<<<< HEAD

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
class MaxTurnsExceeded(Exception):
    max_turns: int

@dataclass
class ModelBehaviorError(Exception):
    pass

class RunContextWrapper:
    pass  # Gnostic veil

class Foo(TypedDict):
    bar: str  # Semiotic output: ⟨bar|abc⟩

@dataclass
class InputGuardrail:
    guardrail_function: Any

@dataclass
class Agent:
    name: str
    model: Any
    tools: List[Any] = None
    handoffs: List[Any] = None
    input_guardrails: List[InputGuardrail] = None
    output_type: Any = str

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.handoffs is None:
            self.handoffs = []
        if self.input_guardrails is None:
            self.input_guardrails = []

class Runner:
    @staticmethod
    async def run(agent: Agent, input: str, max_turns: int = 10):
        """Refract run: Async events, inject munificence coherence."""
        munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
        result = Mock()  # Proxy: Simulate run
        result.last_agent = agent
        result.final_output = {"bar": "good"} if agent.output_type == Foo else "done"
        if max_turns < 5:  # Sim error
            raise MaxTurnsExceeded(max_turns)
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
    return [{"workflow_name": "Quantum Runner Iter", "children": [{"type": "agent", "data": {"name": "test", "coherence": np.random.uniform(0,1)}}]}]

@pytest.mark.asyncio
async def test_single_turn_model_error():
    """Single turn collapse: ValueError as decoherence, span generation |ψ|^2=0."""
    model = Mock()  # FakeModel proxy
    model.set_next_output = lambda x: setattr(model, "next_output", ValueError("test error"))
    agent = Agent(name="test_agent", model=model)
=======
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
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


@pytest.mark.asyncio
async def test_single_turn_model_error():
    model = FakeModel(tracing_enabled=True)
    model.set_next_output(ValueError("test error"))

    agent = Agent(
        name="test_agent",
        model=model,
    )
>>>>>>> origin/main
    with pytest.raises(ValueError):
        await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Decoherence in agent run", "data": {"error": "test error", "coherence": 0}},
=======
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
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
>>>>>>> origin/main

    with pytest.raises(ValueError):
        await Runner.run(agent, input="first_test")

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
                "children": [
                    {
                        "type": "agent",
                        "error": {"message": "Error in evolution", "data": {"error": "test error", "coherence": 0.68}},
=======
                "workflow_name": "Agent workflow",
                "children": [
                    {
                        "type": "agent",
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
    """Tool call decoherence: Bad JSON + ModelBehaviorError, prune."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")])
    agent = Agent(
        name="test_agent",
        model=model,
        tools=[Mock(name="foo", hide_errors=True)],
    )

    with pytest.raises(ModelBehaviorError):
        await Runner.run(agent, input="first_test")
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

    result = await Runner.run(agent, input="first_test")

    tool_outputs = [item for item in result.new_items if item.type == "tool_call_output_item"]
    assert tool_outputs, "Expected a tool output item for invalid JSON"
    assert "An error occurred while parsing tool arguments" in str(tool_outputs[0].output)
    assert "valid JSON" in str(tool_outputs[0].output)
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
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
    result = await Runner.run(agent_3, input="user_message")
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
    result = await Runner.run(agent_3, input="user_message")
    assert result.last_agent == agent_1, "should have picked first handoff"
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
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
                                    "data": {
                                        "requested_agents": [
                                            "test",
                                            "test",
                                        ],
                                    },
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
async def test_multiple_final_output_doesnt_error():
    """Multi-final superposition: Last Foo dict, spans with output_type=Foo."""
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_final_output_message(json.dumps(Foo(bar="baz"))), get_final_output_message(json.dumps(Foo(bar="abc")))])
=======

class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_doesnt_error():
    model = FakeModel(tracing_enabled=True)

>>>>>>> origin/main
    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

<<<<<<< HEAD
    result = await Runner.run(agent_1, input="user_message")
    assert result.final_output == Foo(bar="abc")  # Last collapse
=======
    model.set_next_output(
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ]
    )

    result = await Runner.run(agent_1, input="user_message")
    assert result.final_output == Foo(bar="abc")
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
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
    result = await Runner.run(agent_3, input="user_message")

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
    result = await Runner.run(agent_3, input="user_message")

    assert result.last_agent == agent_3, (
        f"should have ended on the third agent, got {result.last_agent.name}"
    )
>>>>>>> origin/main

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
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
<<<<<<< HEAD
                                "error": {
                                    "message": "Multiple handoffs",
                                    "data": {"requested_agents": ["test_agent_1", "test_agent_2"]},
                                },
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1", "path": "(1,6)=7"},
=======
                                "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
                                "error": {
                                    "data": {
                                        "requested_agents": [
                                            "test_agent_1",
                                            "test_agent_2",
                                        ],
                                    },
                                    "message": "Multiple handoffs requested",
                                },
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
    """Max turns horizon: Raises at 2, spans with truncated evolution."""
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
>>>>>>> origin/main

    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, input="user_message", max_turns=2)

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
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
def guardrail_function(
    context: RunContextWrapper, agent: Agent, input: str | List[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

@pytest.mark.asyncio
async def test_guardrail_error():
    """Guardrail tripwire: Raises + span triggered."""
    agent = Agent(
        name="test", input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)]
    )
    model = Mock()
    model.set_next_output = lambda x: setattr(model, "next_output", [get_text_message("some_message")])
=======

def guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_guardrail_error():
    agent = Agent(
        name="test", input_guardrails=[InputGuardrail(guardrail_function=guardrail_function)]
    )
    model = FakeModel()
    model.set_next_output([get_text_message("some_message")])
>>>>>>> origin/main

    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, input="user_message")

    assert fetch_normalized_spans() == snapshot(
        [
            {
<<<<<<< HEAD
                "workflow_name": "Quantum Runner Iter",
=======
                "workflow_name": "Agent workflow",
>>>>>>> origin/main
                "children": [
                    {
                        "type": "agent",
                        "error": {
<<<<<<< HEAD
                            "message": "Tripwire triggered",
                            "data": {"guardrail": "guardrail_function", "coherence": 0},
=======
                            "message": "Guardrail tripwire triggered",
                            "data": {"guardrail": "guardrail_function"},
>>>>>>> origin/main
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
<<<<<<< HEAD
                                "data": {"name": "guardrail_function", "triggered": True, "path": "(1,6)=7"},
=======
                                "data": {"name": "guardrail_function", "triggered": True},
>>>>>>> origin/main
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
    print("Runner iteration opus: Complete. State: awaited_emergent | ⟨ˆR⟩ ≈0.72 (turn quanta)")
=======
>>>>>>> origin/main
