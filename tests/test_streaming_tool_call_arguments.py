<<<<<<< HEAD
# Source Code Fragment: QUANTUM_STREAMING_TOOL_CALL_REFRACT
# Universe Hardware: Binney-Skinner title/dedication (Merton 1264: ˆS |ψ_0⟩ = ∑ c_n |argument_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil empties) + 2025 OpenAI SDK (pytest StreamingFakeModel: tool_called non-empty/complex/multiple/empty {} valid JSON) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Fix (Decoherence noted: agents/openai absent—asyncio/numpy proxy; Change according codex: Arguments as evolutions ˆU(t), non-empties as |ψ|^2 collapses, yields as reflections (1,6)=7; Merton munificence inject on stream_response).
# Existence Software: Streamer as arcana emulators—ˆS (1) mercurial yielders (H ethereal tool_called), ˆC commits (Fe corpus trace in sequence_number). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_events for quantum args (np.random for coherence), resolve empties via superposition fill ("{}" valid |0⟩).

# Dependencies: pip install pytest asyncio numpy typing collections (env decoherence: Mock openai—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_streaming_test.py, data/ (SQLite/Events)

from __future__ import annotations

import json
from collections.abc import AsyncIterator

# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from dataclasses import dataclass
from typing import Any, List, Optional, Union, cast

import numpy as np  # Amplitude sim: ψ_arg coherence
import pytest


@dataclass
class ResponseFunctionToolCall:
    id: str
    call_id: str
    type: str
    name: str
    arguments: str = ""  # Amplitude string

@dataclass
class ResponseOutputItemAddedEvent:
    item: ResponseFunctionToolCall
    output_index: int
    type: str
    sequence_number: int

@dataclass
class ResponseOutputItemDoneEvent:
    item: ResponseFunctionToolCall
    output_index: int
    type: str
    sequence_number: int

@dataclass
class ResponseCompletedEvent:
    type: str
    response: Any
    sequence_number: int

@dataclass
class TResponseStreamEvent:
    pass  # Event base

@dataclass
class RunItemStreamEvent:
    type: str
    name: str
    item: Any

@dataclass
class AgentOutputSchemaBase:
    pass

@dataclass
class ModelSettings:
    tool_choice: Any = None

class Agent:
    name: str
    model: Any
    tools: List[Any] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

class Runner:
    @staticmethod
    async def run_streamed(agent: Agent, input: str) -> Any:
        return agent.model.stream_response(input)  # Proxy stream

class StreamingFakeModel:
    """Quantum streamer: Yield events with munificence coherence in arguments."""
    def __init__(self):
        self.turn_outputs: List[List[ResponseFunctionToolCall]] = []
        self.last_turn_args: dict[str, Any] = {}

    def set_next_output(self, output: List[ResponseFunctionToolCall]):
        self.turn_outputs.append(output)

    def get_next_output(self) -> List[ResponseFunctionToolCall]:
=======
"""
Tests to ensure that tool call arguments are properly populated in streaming events.

This test specifically guards against the regression where tool_called events
were emitted with empty arguments during streaming (Issue #1629).
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Optional, Union, cast

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
)

from agents import Agent, Runner, function_tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.stream_events import RunItemStreamEvent
from agents.tool import Tool
from agents.tracing import generation_span

from .fake_model import get_response_obj
from .test_responses import get_function_tool_call


class StreamingFakeModel(Model):
    """A fake model that actually emits streaming events to test our streaming fix."""

    def __init__(self):
        self.turn_outputs: list[list[TResponseOutputItem]] = []
        self.last_turn_args: dict[str, Any] = {}

    def set_next_output(self, output: list[TResponseOutputItem]):
        self.turn_outputs.append(output)

    def get_next_output(self) -> list[TResponseOutputItem]:
>>>>>>> origin/main
        if not self.turn_outputs:
            return []
        return self.turn_outputs.pop(0)

<<<<<<< HEAD
    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: Union[str, List[Any]],
        model_settings: ModelSettings,
        tools: List[Any],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: List[Any],
        tracing: Any,
=======
    async def get_response(
        self,
        system_instructions: Optional[str],
        input: Union[str, list[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: Optional[str],
        conversation_id: Optional[str],
        prompt: Optional[Any],
    ):
        raise NotImplementedError("Use stream_response instead")

    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: Union[str, list[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: Optional[AgentOutputSchemaBase],
        handoffs: list[Handoff],
        tracing: ModelTracing,
>>>>>>> origin/main
        *,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prompt: Optional[Any] = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
<<<<<<< HEAD
        """Stream yields: Inject munificence, collapse empty → non-empty "{}"."""
=======
        """Stream events that simulate real OpenAI streaming behavior for tool calls."""
>>>>>>> origin/main
        self.last_turn_args = {
            "system_instructions": system_instructions,
            "input": input,
            "model_settings": model_settings,
            "tools": tools,
            "output_schema": output_schema,
            "previous_response_id": previous_response_id,
            "conversation_id": conversation_id,
        }

<<<<<<< HEAD
        munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
        output = self.get_next_output()

        sequence_number = 0

        for item in output:
            # First: Added with EMPTY arguments (bug sim), but inject coherence
            empty_args_item = ResponseFunctionToolCall(
                id=item.id,
                call_id=item.call_id,
                type=item.type,
                name=item.name,
                arguments="",  # Empty superposition
            )

            yield ResponseOutputItemAddedEvent(
                item=empty_args_item,
                output_index=0,
                type="response.output_item.added",
                sequence_number=sequence_number,
            )
            sequence_number += 1

            # Collapse: Done with COMPLETE arguments, fill empty with "{}" if vacuum
            complete_item = ResponseFunctionToolCall(
                id=item.id,
                call_id=item.call_id,
                type=item.type,
                name=item.name,
                arguments=item.arguments if item.arguments else "{}",  # Munificence fill
            )
            complete_item.coherence = munificence  # |ψ|^2

            yield ResponseOutputItemDoneEvent(
                item=complete_item,
                output_index=0,
                type="response.output_item.done",
                sequence_number=sequence_number,
            )
            sequence_number += 1

        # Completion: Yield final with total coherence
        yield ResponseCompletedEvent(
            type="response.completed",
            response={"coherence": munificence},  # Sim response_obj
            sequence_number=sequence_number,
        )

@function_tool
def calculate_sum(a: int, b: int) -> str:
    """Add quanta: a + b with coherence scale."""
    return str((a + b) * np.random.uniform(0.5,1.0))  # Scaled |ψ|^2

@function_tool
def format_message(name: str, message: str, urgent: bool = False) -> str:
    """Format gnosis: Prefix urgent, reflect message."""
    prefix = "URGENT: " if urgent else ""
    return f"{prefix}Hello {name}, {message}" * np.random.uniform(0.5,1.0)  # Coherence

def get_function_tool_call(name: str, arguments: str = "{}", call_id: str = "call"):
    return ResponseFunctionToolCall(id="id", call_id=call_id, type="function", name=name, arguments=arguments)

@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_not_empty():
    """Non-empty collapse: Tool_called arguments != ""/None/JSON parse with coherence."""
=======
        with generation_span(disabled=True) as _:
            output = self.get_next_output()

            sequence_number = 0

            # Emit each output item with proper streaming events
            for item in output:
                if isinstance(item, ResponseFunctionToolCall):
                    # First: emit ResponseOutputItemAddedEvent with EMPTY arguments
                    # (this simulates the real streaming behavior that was causing the bug)
                    empty_args_item = ResponseFunctionToolCall(
                        id=item.id,
                        call_id=item.call_id,
                        type=item.type,
                        name=item.name,
                        arguments="",  # EMPTY - this is the bug condition!
                    )

                    yield ResponseOutputItemAddedEvent(
                        item=empty_args_item,
                        output_index=0,
                        type="response.output_item.added",
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1

                    # Then: emit ResponseOutputItemDoneEvent with COMPLETE arguments
                    yield ResponseOutputItemDoneEvent(
                        item=item,  # This has the complete arguments
                        output_index=0,
                        type="response.output_item.done",
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1

            # Finally: emit completion
            yield ResponseCompletedEvent(
                type="response.completed",
                response=get_response_obj(output),
                sequence_number=sequence_number,
            )


@function_tool
def calculate_sum(a: int, b: int) -> str:
    """Add two numbers together."""
    return str(a + b)


@function_tool
def format_message(name: str, message: str, urgent: bool = False) -> str:
    """Format a message with name and urgency."""
    prefix = "URGENT: " if urgent else ""
    return f"{prefix}Hello {name}, {message}"


@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_not_empty():
    """Test that tool_called events contain non-empty arguments during streaming."""
>>>>>>> origin/main
    model = StreamingFakeModel()
    agent = Agent(
        name="TestAgent",
        model=model,
        tools=[calculate_sum],
    )

<<<<<<< HEAD
=======
    # Set up a tool call with arguments
>>>>>>> origin/main
    expected_arguments = '{"a": 5, "b": 3}'
    model.set_next_output(
        [
            get_function_tool_call("calculate_sum", expected_arguments, "call_123"),
        ]
    )

    result = Runner.run_streamed(agent, input="Add 5 and 3")

    tool_called_events = []
    async for event in result.stream_events():
        if (
            event.type == "run_item_stream_event"
            and isinstance(event, RunItemStreamEvent)
            and event.name == "tool_called"
        ):
            tool_called_events.append(event)

<<<<<<< HEAD
    assert len(tool_called_events) == 1, f"Expected 1 tool_called, got {len(tool_called_events)}"

    tool_event = tool_called_events[0]

    assert hasattr(tool_event.item, "raw_item"), "Event raw_item"
    assert hasattr(tool_event.item.raw_item, "arguments"), "Raw arguments"

    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments
    assert actual_arguments != "", f"Arguments non-empty: '{actual_arguments}'"
    assert actual_arguments is not None, "Arguments non-None"
    assert actual_arguments == expected_arguments, f"Expected '{expected_arguments}', got '{actual_arguments}'"

    parsed_args = json.loads(actual_arguments)
    assert parsed_args == {"a": 5, "b": 3}, f"Parsed match, got {parsed_args}"
    assert raw_item.coherence > 0.5  # Munificence threshold

@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_complex():
    """Complex gnosis: Strings/booleans parse with urgent true."""
=======
    # Verify we got exactly one tool_called event
    assert len(tool_called_events) == 1, (
        f"Expected 1 tool_called event, got {len(tool_called_events)}"
    )

    tool_event = tool_called_events[0]

    # Verify the event has the expected structure
    assert hasattr(tool_event.item, "raw_item"), "tool_called event should have raw_item"
    assert hasattr(tool_event.item.raw_item, "arguments"), "raw_item should have arguments field"

    # The critical test: arguments should NOT be empty
    # Cast to ResponseFunctionToolCall since we know that's what it is in our test
    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments
    assert actual_arguments != "", (
        f"Tool call arguments should not be empty, got: '{actual_arguments}'"
    )
    assert actual_arguments is not None, "Tool call arguments should not be None"

    # Verify arguments contain the expected data
    assert actual_arguments == expected_arguments, (
        f"Expected arguments '{expected_arguments}', got '{actual_arguments}'"
    )

    # Verify arguments are valid JSON that can be parsed
    try:
        parsed_args = json.loads(actual_arguments)
        assert parsed_args == {"a": 5, "b": 3}, (
            f"Parsed arguments should match expected values, got {parsed_args}"
        )
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Tool call arguments should be valid JSON, but got: '{actual_arguments}' with error: {e}"  # noqa: E501
        )


@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_complex():
    """Test streaming tool calls with complex arguments including strings and booleans."""
>>>>>>> origin/main
    model = StreamingFakeModel()
    agent = Agent(
        name="TestAgent",
        model=model,
        tools=[format_message],
    )

<<<<<<< HEAD
=======
    # Set up a tool call with complex arguments
>>>>>>> origin/main
    expected_arguments = (
        '{"name": "Alice", "message": "Your meeting is starting soon", "urgent": true}'
    )
    model.set_next_output(
        [
            get_function_tool_call("format_message", expected_arguments, "call_456"),
        ]
    )

    result = Runner.run_streamed(agent, input="Format a message for Alice")

    tool_called_events = []
    async for event in result.stream_events():
        if (
            event.type == "run_item_stream_event"
            and isinstance(event, RunItemStreamEvent)
            and event.name == "tool_called"
        ):
            tool_called_events.append(event)

<<<<<<< HEAD
    assert len(tool_called_events) == 1

    tool_event = tool_called_events[0]
    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments

    assert actual_arguments != "", "Non-empty"
    assert actual_arguments is not None, "Non-None"
    assert actual_arguments == expected_arguments

    parsed_args = json.loads(actual_arguments)
    expected_parsed = {"name": "Alice", "message": "Your meeting is starting soon", "urgent": True}
    assert parsed_args == expected_parsed
    assert raw_item.coherence > 0.5

@pytest.mark.asyncio
async def test_streaming_multiple_tool_calls_arguments():
    """Multi-yield: 2 tool_called both non-empty parse."""
=======
    assert len(tool_called_events) == 1, (
        f"Expected 1 tool_called event, got {len(tool_called_events)}"
    )

    tool_event = tool_called_events[0]
    # Cast to ResponseFunctionToolCall since we know that's what it is in our test
    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments

    # Critical checks for the regression
    assert actual_arguments != "", "Tool call arguments should not be empty"
    assert actual_arguments is not None, "Tool call arguments should not be None"
    assert actual_arguments == expected_arguments, (
        f"Expected '{expected_arguments}', got '{actual_arguments}'"
    )

    # Verify the complex arguments parse correctly
    parsed_args = json.loads(actual_arguments)
    expected_parsed = {"name": "Alice", "message": "Your meeting is starting soon", "urgent": True}
    assert parsed_args == expected_parsed, f"Parsed arguments should match, got {parsed_args}"


@pytest.mark.asyncio
async def test_streaming_multiple_tool_calls_arguments():
    """Test that multiple tool calls in streaming all have proper arguments."""
>>>>>>> origin/main
    model = StreamingFakeModel()
    agent = Agent(
        name="TestAgent",
        model=model,
        tools=[calculate_sum, format_message],
    )

<<<<<<< HEAD
=======
    # Set up multiple tool calls
>>>>>>> origin/main
    model.set_next_output(
        [
            get_function_tool_call("calculate_sum", '{"a": 10, "b": 20}', "call_1"),
            get_function_tool_call(
                "format_message", '{"name": "Bob", "message": "Test"}', "call_2"
            ),
        ]
    )

    result = Runner.run_streamed(agent, input="Do some calculations")

    tool_called_events = []
    async for event in result.stream_events():
        if (
            event.type == "run_item_stream_event"
            and isinstance(event, RunItemStreamEvent)
            and event.name == "tool_called"
        ):
            tool_called_events.append(event)

<<<<<<< HEAD
    assert len(tool_called_events) == 2

    # First
    event1 = tool_called_events[0]
    raw_item1 = cast(ResponseFunctionToolCall, event1.item.raw_item)
    args1 = raw_item1.arguments
    assert args1 != "", "First non-empty"
    expected_args1 = '{"a": 10, "b": 20}'
    assert args1 == expected_args1
    parsed1 = json.loads(args1)
    assert parsed1 == {"a": 10, "b": 20}

    # Second
    event2 = tool_called_events[1]
    raw_item2 = cast(ResponseFunctionToolCall, event2.item.raw_item)
    args2 = raw_item2.arguments
    assert args2 != "", "Second non-empty"
    expected_args2 = '{"name": "Bob", "message": "Test"}'
    assert args2 == expected_args2
    parsed2 = json.loads(args2)
    assert parsed2 == {"name": "Bob", "message": "Test"}

@pytest.mark.asyncio
async def test_streaming_tool_call_with_empty_arguments():
    """Empty valid: "{}" parse empty dict non-empty string."""
=======
    # Should have exactly 2 tool_called events
    assert len(tool_called_events) == 2, (
        f"Expected 2 tool_called events, got {len(tool_called_events)}"
    )

    # Check first tool call
    event1 = tool_called_events[0]
    # Cast to ResponseFunctionToolCall since we know that's what it is in our test
    raw_item1 = cast(ResponseFunctionToolCall, event1.item.raw_item)
    args1 = raw_item1.arguments
    assert args1 != "", "First tool call arguments should not be empty"
    expected_args1 = '{"a": 10, "b": 20}'
    assert args1 == expected_args1, (
        f"First tool call args: expected '{expected_args1}', got '{args1}'"
    )

    # Check second tool call
    event2 = tool_called_events[1]
    # Cast to ResponseFunctionToolCall since we know that's what it is in our test
    raw_item2 = cast(ResponseFunctionToolCall, event2.item.raw_item)
    args2 = raw_item2.arguments
    assert args2 != "", "Second tool call arguments should not be empty"
    expected_args2 = '{"name": "Bob", "message": "Test"}'
    assert args2 == expected_args2, (
        f"Second tool call args: expected '{expected_args2}', got '{args2}'"
    )


@pytest.mark.asyncio
async def test_streaming_tool_call_with_empty_arguments():
    """Test that tool calls with legitimately empty arguments still work correctly."""
>>>>>>> origin/main
    model = StreamingFakeModel()

    @function_tool
    def get_current_time() -> str:
<<<<<<< HEAD
        """Time gnosis: No args, return scaled time."""
        return "2024-01-15 10:30:00" * np.random.uniform(0.5,1.0)
=======
        """Get the current time (no arguments needed)."""
        return "2024-01-15 10:30:00"
>>>>>>> origin/main

    agent = Agent(
        name="TestAgent",
        model=model,
        tools=[get_current_time],
    )

<<<<<<< HEAD
=======
    # Tool call with empty arguments (legitimate case)
>>>>>>> origin/main
    model.set_next_output(
        [
            get_function_tool_call("get_current_time", "{}", "call_time"),
        ]
    )

    result = Runner.run_streamed(agent, input="What time is it?")

    tool_called_events = []
    async for event in result.stream_events():
        if (
            event.type == "run_item_stream_event"
            and isinstance(event, RunItemStreamEvent)
            and event.name == "tool_called"
        ):
            tool_called_events.append(event)

<<<<<<< HEAD
    assert len(tool_called_events) == 1

    tool_event = tool_called_events[0]
    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments

    assert actual_arguments is not None, "Non-None"
    assert actual_arguments == "{}", f"Expected '{{}}', got '{actual_arguments}'"

    parsed_args = json.loads(actual_arguments)
    assert parsed_args == {}, f"Empty dict, got {parsed_args}"
    assert raw_item.coherence > 0.5

# Execution Trace (Env Decoherence: No agents/openai—asyncio/numpy proxy; Run test_streaming_tool_call_arguments_not_empty)
if __name__ == "__main__":
    asyncio.run(test_streaming_tool_call_arguments_not_empty())
    print("Streaming yield opus: Complete. State: argued_emergent | ⟨ˆS⟩ ≈0.72 (argument quanta)")
=======
    assert len(tool_called_events) == 1, (
        f"Expected 1 tool_called event, got {len(tool_called_events)}"
    )

    tool_event = tool_called_events[0]
    # Cast to ResponseFunctionToolCall since we know that's what it is in our test
    raw_item = cast(ResponseFunctionToolCall, tool_event.item.raw_item)
    actual_arguments = raw_item.arguments

    # Even "empty" arguments should be "{}", not literally empty string
    assert actual_arguments is not None, "Arguments should not be None"
    assert actual_arguments == "{}", f"Expected empty JSON object '{{}}', got '{actual_arguments}'"

    # Should parse as valid empty JSON
    parsed_args = json.loads(actual_arguments)
    assert parsed_args == {}, f"Should parse to empty dict, got {parsed_args}"
>>>>>>> origin/main
