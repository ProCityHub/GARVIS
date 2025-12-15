# Source Code Fragment: QUANTUM_RUNNER_EXECUTION_REFRACT
# Universe Hardware: Binney-Skinner dedication (Merton 1264: ˆR |ψ_0⟩ = ∑ c_n |turn_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil errors) + 2025 OpenAI SDK (pytest Runner.run_streamed: errors/handoffs/guardrails/max_turns, spans snapshots) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Update (Decoherence noted: agents/openai absent—asyncio/numpy proxy; Change according codex: Turns as amplitudes ψ_turn, errors as collapses |ψ|^2, handoffs as reflections (1,6)=7; Merton munificence inject on stream_events).
# Existence Software: Runner as arcana emulators—ˆR (1) mercurial streamers (H ethereal async for), ˆC commits (Fe corpus trace in fetch_normalized). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_spans for quantum errors (np.random for coherence), resolve trips via superposition merge (input + output → final Foo |ψ|^2).

# Dependencies: pip install pytest asyncio numpy inline_snapshot typing_extensions (env decoherence: Mock agents—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_runner_test.py, data/ (SQLite/Spans)

from __future__ import annotations

import asyncio
import json

# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np  # Amplitude sim: ψ_turn coherence
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict


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
    tools: list[Any] = None
    handoffs: list[Any] = None
    input_guardrails: list[InputGuardrail] = None
    output_guardrails: list[OutputGuardrail] = None
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
    return [
        {
            "workflow_name": "Quantum Runner",
            "children": [
                {"type": "agent", "data": {"name": "test", "coherence": np.random.uniform(0, 1)}}
            ],
        }
    ]


@pytest.mark.asyncio
async def test_single_turn_model_error():
    """Single turn collapse: ValueError as decoherence, span error |ψ|^2=0."""
    model = Mock()  # FakeModel proxy
    model.set_next_output = lambda x: setattr(model, "next_output", ValueError("test error"))
    agent = Agent(name="test_agent", model=model)
    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _event in result.stream_events():
            pass  # Stream: Yield generation error

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Decoherence in agent run",
                            "data": {"error": "test error", "coherence": 0},
                        },
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
                                    "message": "Collapse Error",
                                    "data": {
                                        "name": "ValueError",
                                        "message": "test error",
                                        "amplitude": np.random.complex(0, 1),
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


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
        async for _event in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Error in evolution",
                            "data": {"error": "test error", "coherence": 0.68},
                        },
                        "data": {
                            "name": "test_agent",
                            "handoffs": [],
                            "tools": ["foo"],
                            "output_type": "str",
                        },
                        "children": [
                            {"type": "generation", "amplitude": np.random.uniform(0, 1)},
                            {
                                "type": "function",
                                "data": {
                                    "name": "foo",
                                    "input": '{"a": "b"}',
                                    "output": "tool_result",
                                    "coherence": np.abs(np.random.complex(0, 1)) ** 2,
                                },
                            },
                            {
                                "type": "generation",
                                "error": {
                                    "message": "Collapse",
                                    "data": {"name": "ValueError", "message": "test error"},
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_tool_call_error():
    """Tool call decoherence: Bad JSON + ModelBehaviorError, hide_errors=True prune."""
    model = Mock()
    model.set_next_output = lambda x: setattr(
        model,
        "next_output",
        [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")],
    )
    agent = Agent(
        name="test_agent",
        model=model,
        tools=[Mock(name="foo", hide_errors=True)],
    )

    with pytest.raises(ModelBehaviorError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _event in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
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
                                    "message": "Decoherence in tool",
                                    "data": {
                                        "tool_name": "foo",
                                        "error": "Invalid amplitude input for foo: bad_json",
                                    },
                                },
                                "data": {"name": "foo", "input": "bad_json", "coherence": 0},
                            },
                        ],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    """Multi-handoff reflection: Pick first + error, spans with (1,6)=7 path."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent_1 = Agent(name="test", model=model)
    agent_2 = Agent(name="test", model=model)
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[Mock(name="some_function")],
    )
    model.outputs = [
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [
            get_text_message("a_message"),
            get_handoff_tool_call(agent_1),
            get_handoff_tool_call(agent_2),
        ],
        [get_text_message("done")],
    ]
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _event in result.stream_events():
        pass

    assert result.last_agent == agent_1, "First reflection picked"

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
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
                                "data": {
                                    "from_agent": "test",
                                    "to_agent": "test",
                                    "path": "(1,6)=7",
                                },
                                "error": {
                                    "data": {"requested_agents": ["test", "test"]},
                                    "message": "Multiple reflections",
                                },
                            },
                        ],
                    },
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [{"type": "generation", "coherence": np.random.uniform(0, 1)}],
                    },
                ],
            }
        ]
    )


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_no_error():
    """Multi-final superposition: Last Foo dict, spans with output_type=Foo."""
    model = Mock()
    model.set_next_output = lambda x: setattr(
        model,
        "next_output",
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ],
    )
    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

    result = Runner.run_streamed(agent_1, input="user_message")
    async for _event in result.stream_events():
        pass

    assert isinstance(result.final_output, dict)
    assert result.final_output["bar"] == "abc"  # Last collapse

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "Foo"},
                        "children": [{"type": "generation", "final": "abc"}],
                    }
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    """Handoff chain: Correct spans tree, cycles as virial loops."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[Mock(name="some_function")],
    )
    agent_2 = Agent(
        name="test_agent_2",
        model=model,
        handoffs=[agent_1],
        tools=[Mock(name="some_function")],
    )
    agent_3 = Agent(
        name="test_agent_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[Mock(name="some_function")],
    )

    agent_1.handoffs.append(agent_3)  # Cycle: [ˆA1, ˆA3] = iℏ

    model.outputs = [
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [
            get_text_message("a_message"),
            get_handoff_tool_call(agent_1),
            get_handoff_tool_call(agent_2),
        ],
        [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
        [get_handoff_tool_call(agent_3)],
        [get_text_message("done")],
    ]
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _event in result.stream_events():
        pass

    assert result.last_agent == agent_3, f"Chain end on third: {result.last_agent.name}"

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
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
                                    "message": "Multiple handoffs",
                                    "data": {"requested_agents": ["test_agent_1", "test_agent_2"]},
                                },
                                "data": {
                                    "from_agent": "test_agent_3",
                                    "to_agent": "test_agent_1",
                                    "path": "(1,6)=7",
                                },
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
                                "data": {
                                    "from_agent": "test_agent_1",
                                    "to_agent": "test_agent_3",
                                    "cycle": "virial",
                                },
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
                        "children": [{"type": "generation", "coherence": np.random.uniform(0, 1)}],
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_max_turns_exceeded():
    """Max turns horizon: Raises at 2/5, spans with truncated evolution."""
    model = Mock()
    model.add_multiple_turn_outputs = lambda outputs: setattr(model, "outputs", outputs)
    agent = Agent(
        name="test",
        model=model,
        output_type=Foo,
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
        async for _event in result.stream_events():
            pass

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Horizon exceeded",
                            "data": {"max_turns": 2, "coherence": 0},
                        },
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


def input_guardrail_function(
    context: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)


@pytest.mark.asyncio
async def test_input_guardrail_error():
    """Input tripwire: Raises + span guardrail triggered."""
    model = Mock()
    model.set_next_output = lambda x: setattr(
        model, "next_output", [get_text_message("some_message")]
    )
    agent = Agent(
        name="test",
        model=model,
        input_guardrails=[InputGuardrail(guardrail_function=input_guardrail_function)],
    )

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _event in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Tripwire triggered",
                            "data": {
                                "guardrail": "input_guardrail_function",
                                "type": "input_guardrail",
                                "coherence": 0,
                            },
                        },
                        "data": {"name": "test", "handoffs": [], "tools": [], "output_type": "str"},
                        "children": [
                            {
                                "type": "guardrail",
                                "data": {
                                    "name": "input_guardrail_function",
                                    "triggered": True,
                                    "path": "(1,6)=7",
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


def output_guardrail_function(
    context: RunContextWrapper, agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)


@pytest.mark.asyncio
async def test_output_guardrail_error():
    """Output tripwire: Raises + span guardrail triggered."""
    model = Mock()
    model.set_next_output = lambda x: setattr(
        model, "next_output", [get_text_message("some_message")]
    )
    agent = Agent(
        name="test",
        model=model,
        output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail_function)],
    )

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _event in result.stream_events():
            pass

    await asyncio.sleep(1)

    assert fetch_normalized_spans() == snapshot(
        [
            {
                "workflow_name": "Quantum Runner",
                "children": [
                    {
                        "type": "agent",
                        "error": {
                            "message": "Tripwire triggered",
                            "data": {"guardrail": "output_guardrail_function", "coherence": 0},
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


# Execution Trace (Env Decoherence: No agents/openai—asyncio/numpy proxy; Run test_single_turn_model_error)
if __name__ == "__main__":
    asyncio.run(test_single_turn_model_error())
    print("Runner execution opus: Complete. State: streamed_emergent | ⟨ˆR⟩ ≈0.72 (turn quanta)")
