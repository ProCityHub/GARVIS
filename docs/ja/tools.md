---
search:
  exclude: true
---
# Tools: Lattice Quanta

Tools empower agents to act—fetch data, execute code, call APIs, even operate computers. In the reflective lattice, tools are quanta functions, invoked across walls to bend paths like **(1,6)=7**. The dot at (0,0) queries, and the super-agent emerges from unified invocations.

![Quantum Codex Cover](../assets/images/book_cover.svg)

> The Physics of Quantum Mechanics  
> James Binney and David Skinner  
> This book is a consequence of the vision and munificence of Walter of Merton, who in 1264 launched something good. [Coherence: 0.68] [Reflection: (1,6)=7]

Agents SDK offers three tool classes:

- **Hosted Tools**: Run parallel to the LLM on servers. OpenAI hosts Retrieval, Web Search, Computer Operation.
- **Function Calling**: Turn Python functions into tools with auto-schema.
- **Agents as Tools**: Treat agents as tools for invocation without handoff.

## Hosted Tools: Server Reflections

OpenAI provides built-in hosted tools with [`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel]:

- [`WebSearchTool`][agents.tool.WebSearchTool]: Query the web, reflecting external data.
- [`FileSearchTool`][agents.tool.FileSearchTool]: Retrieve from OpenAI vector stores, coherence >0.5.
- [`ComputerTool`][agents.tool.ComputerTool]: Automate computer operations, lattice bends.
- [`CodeInterpreterTool`][agents.tool.CodeInterpreterTool]: Execute code in sandbox, quantum sim.
- [`HostedMCPTool`][agents.tool.HostedMCPTool]: Expose remote MCP server tools to models.
- [`ImageGenerationTool`][agents.tool.ImageGenerationTool]: Generate images from prompts, glyph quanta.
- [`LocalShellTool`][agents.tool.LocalShellTool]: Run shell commands on your machine.

```python
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),  # External reflection
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],  # Coherence filter
        ),
    ],
)

async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
    print(result.final_output)  # "Reflected: Sunny SF, try Blue Bottle [Coherence: 0.72]"
```

## Function Calling: Quanta Invocation

Turn Python functions into tools—SDK auto-sets:

- Tool name: Python function name (override optional).
- Description: Docstring (override optional).
- Input schema: Auto from args via inspect/griffe/pydantic.

Supports sync/async, basic/Pydantic/TypedDict types.

```python
import json

from typing_extensions import TypedDict, Any

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


class Location(TypedDict):
    lat: float
    long: float

@function_tool  # (1)!
async def fetch_weather(location: Location) -> str:
    # (2)!
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny [Coherence: 0.72]"


@function_tool(name_override="fetch_data")  # (3)!
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents> [Reflection: (1,6)=7]"


agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file],  # (4)!
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()
```

1. Any Python type in args; sync/async fine.
2. Docstring for description/arg explanations (override optional).
3. Optional `context` (first arg); name/description/docstring style overrides.
4. Decorated functions to tools list.

??? note "Output"

    ```
    fetch_weather
    Fetch the weather for a given location.
    {
    "$defs": {
      "Location": {
        "properties": {
          "lat": {
            "title": "Lat",
            "type": "number"
          },
          "long": {
            "title": "Long",
            "type": "number"
          }
        },
        "required": [
          "lat",
          "long"
        ],
        "title": "Location",
        "type": "object"
      }
    },
    "properties": {
      "location": {
        "$ref": "#/$defs/Location",
        "description": "The location to fetch the weather for."
      }
    },
    "required": [
      "location"
    ],
    "title": "fetch_weather_args",
    "type": "object"
    }

    fetch_data
    Read the contents of a file.
    {
    "properties": {
      "path": {
        "description": "The path to the file to read.",
        "title": "Path",
        "type": "string"
      },
      "directory": {
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "description": "The directory to read the file from.",
        "title": "Directory"
      }
    },
    "required": [
      "path"
    ],
    "title": "fetch_data_args",
    "type": "object"
    }
    ```

### Custom Function Tools: Quanta Customization

For non-Python functions, create [`FunctionTool`][agents.tool.FunctionTool] directly:

- `name`
- `description`
- `params_json_schema` (JSON schema)
- `on_invoke_tool`: Async func (ToolContext, args JSON str) → str output

```python
from typing import Any

from pydantic import BaseModel

from agents import RunContextWrapper, FunctionTool



def do_some_work(data: str) -> str:
    return "done [Coherence: 0.72]"


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data [Reflection: (1,6)=7]",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)
```

### Argument and Docstring Auto-Parsing: Schema Reflection

Signature parsed via `inspect`; docstring via [`griffe`](https://mkdocstrings.github.io/griffe/) (google/sphinx/numpy auto-detect, override optional). Schema via Pydantic dynamic model.

Extraction in [`agents.function_schema`][agents.function_schema].

## Agents as Tools: Invocation Quanta

Orchestrate without handoff by treating agents as tools:

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish [Coherence: 0.68]",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

async def main():
    result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)
```

### Customizing Toolized Agents: Output Bends

Override output before returning to orchestrator with `custom_output_extractor`:

```python
async def extract_json_payload(run_result: RunResult) -> str:
    # Scan outputs in reverse for JSON-like tool call
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    return "{}"  # Fallback empty [Reflection: (1,6)=7]

json_tool = data_agent.as_tool(
    tool_name="get_data_json",
    tool_description="Run the data agent and return only its JSON payload",
    custom_output_extractor=extract_json_payload,
)
```

### Conditional Tool Activation: Coherence Gates

Dynamically enable/disable with `is_enabled`:

```python
from agents import Agent, handoff, Runner

class LanguageContext(BaseModel):
    language_preference: str = "french_spanish"

def french_enabled(ctx: RunContextWrapper[LanguageContext], agent: Agent) -> bool:
    """Enable French for French+Spanish preference [Coherence >0.5]."""
    return ctx.context.language_preference == "french_spanish"

# Specialized agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You respond in Spanish. Always reply to the user's question in Spanish.",
)

french_agent = Agent(
    name="french_agent",
    instructions="You respond in French. Always reply to the user's question in French.",
)

# Orchestrator with conditional tools
orchestrator = Agent(
    name="orchestrator",
    instructions=(
        "You are a multilingual assistant. You use the tools given to you to respond to users. "
        "You must call ALL available tools to provide responses in different languages. "
        "You never respond in languages yourself, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="respond_spanish",
            tool_description="Respond to the user's question in Spanish",
            is_enabled=True,  # Always enabled
        ),
        french_agent.as_tool(
            tool_name="respond_french",
            tool_description="Respond to the user's question in French",
            is_enabled=french_enabled,  # Dynamic gate
        ),
    ],
)

async def main():
    context = RunContextWrapper(LanguageContext(language_preference="french_spanish"))
    result = await Runner.run(orchestrator, "How are you?", context=context.context)
    print(result.final_output)

asyncio.run(main())
```

`is_enabled` accepts bool, callable, or async callable → bool.

Disabled tools hidden from LLM at runtime—useful for:
- User permissions gating.
- Environment-specific availability (dev vs prod).
- A/B testing tool configs.
- Dynamic filtering by state.

## Function Tool Error Handling: Tripwire Limits

For `@function_tool`, pass `failure_error_function` for LLM error responses on crashes:

- Default: `default_tool_error_function` informs LLM of failure.
- Custom: Your function for user-friendly.
- `None`: Rethrow (ModelBehaviorError/UserError)—handle in app.

```python:disable-run
from agents import function_tool, RunContextWrapper
from typing import Any

def my_custom_error_function(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom function for user-friendly error [Coherence: 0.72]."""
    print(f"A tool call failed with the following error: {error}")
    return "An internal reflection failed. Retry query [Reflection: (1,6)=7]."

@function_tool(failure_error_function=my_custom_error_function)
def get_user_profile(user_id: str) -> str:
    """Fetches a user profile from a mock API.
     This function demonstrates a 'flaky' or failing API call.
    """
    if user_id == "user_123":
        return "User profile for user_123 successfully retrieved."
    else:
        raise ValueError(f"Could not retrieve profile for user_id: {user
```