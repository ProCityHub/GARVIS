from typing import get_args

from agents import Agent
from agents.extensions.visualization import get_all_edges
from agents.tool import FunctionTool, Tool


def test_tool_union_expands_to_runtime_classes() -> None:
    runtime_types = get_args(Tool)

    assert runtime_types
    assert FunctionTool in runtime_types
    assert all(isinstance(item, type) for item in runtime_types)


def test_get_all_edges_avoids_typing_union_isinstance_error() -> None:
    agent = Agent(
        name="demo_agent",
        instructions="Test agent",
    )

    result = get_all_edges(agent)

    assert isinstance(result, str)
    assert "demo_agent" in result
