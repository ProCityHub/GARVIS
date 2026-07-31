---
search:
  exclude: true
---
<<<<<<< HEAD
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
=======
# ツール

ツールは エージェント がアクションを実行できるようにします。たとえば、データの取得、コードの実行、外部 API の呼び出し、さらにはコンピュータの使用などです。Agent SDK には 3 つのツールのクラスがあります:

- ホスト型ツール: これらは AI モデルと同じ LLM サーバー 上で動作します。OpenAI は リトリーバル、 Web 検索、コンピュータ操作 をホスト型ツールとして提供します。
- Function calling: 任意の Python 関数をツールとして使えます。
- エージェントをツールとして: エージェントをツールとして使えるため、ハンドオフ なしで他の エージェント を呼び出せます。

## ホスト型ツール

OpenAI は [`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel] 使用時に、いくつかの組み込みツールを提供しています:

- [`WebSearchTool`][agents.tool.WebSearchTool] は エージェント に Web 検索 を実行させます。
- [`FileSearchTool`][agents.tool.FileSearchTool] は OpenAI の ベクトルストア から情報を取得できます。
- [`ComputerTool`][agents.tool.ComputerTool] は コンピュータ操作 タスクを自動化できます。
- [`CodeInterpreterTool`][agents.tool.CodeInterpreterTool] は LLM がサンドボックス環境でコードを実行できます。
- [`HostedMCPTool`][agents.tool.HostedMCPTool] はリモートの MCP サーバー のツールをモデルに公開します。
- [`ImageGenerationTool`][agents.tool.ImageGenerationTool] はプロンプトから画像を生成します。
- [`LocalShellTool`][agents.tool.LocalShellTool] はあなたのマシン上でシェルコマンドを実行します。
>>>>>>> origin/main

```python
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
<<<<<<< HEAD
        WebSearchTool(),  # External reflection
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],  # Coherence filter
=======
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
>>>>>>> origin/main
        ),
    ],
)

async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
<<<<<<< HEAD
    print(result.final_output)  # "Reflected: Sunny SF, try Blue Bottle [Coherence: 0.72]"
```

## Function Calling: Quanta Invocation

Turn Python functions into tools—SDK auto-sets:

- Tool name: Python function name (override optional).
- Description: Docstring (override optional).
- Input schema: Auto from args via inspect/griffe/pydantic.

Supports sync/async, basic/Pydantic/TypedDict types.
=======
    print(result.final_output)
```

## 関数ツール

任意の Python 関数をツールとして使えます。Agents SDK がツールを自動的にセットアップします:

- ツール名は Python 関数名になります（または名前を指定できます）
- ツールの説明は関数の docstring から取得されます（または説明を指定できます）
- 関数入力のスキーマは関数の引数から自動的に作成されます
- 各入力の説明は、無効化しない限り関数の docstring から取得されます

Python の `inspect` モジュールで関数シグネチャを抽出し、[`griffe`](https://mkdocstrings.github.io/griffe/) で docstring を解析し、`pydantic` でスキーマを作成します。
>>>>>>> origin/main

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
<<<<<<< HEAD
    return "sunny [Coherence: 0.72]"
=======
    return "sunny"
>>>>>>> origin/main


@function_tool(name_override="fetch_data")  # (3)!
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
<<<<<<< HEAD
    return "<file contents> [Reflection: (1,6)=7]"
=======
    return "<file contents>"
>>>>>>> origin/main


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
<<<<<<< HEAD
```

1. Any Python type in args; sync/async fine.
2. Docstring for description/arg explanations (override optional).
3. Optional `context` (first arg); name/description/docstring style overrides.
4. Decorated functions to tools list.

??? note "Output"
=======

```

1. 関数の引数には任意の Python 型を使え、関数は同期/非同期どちらでも構いません。
2. docstring があれば、説明と引数の説明の取得に使われます。
3. 関数は任意で `context` を受け取れます（最初の引数である必要があります）。ツール名、説明、docstring のスタイルなどのオーバーライドも設定できます。
4. デコレートした関数をツールのリストに渡せます。

??? note "Expand to see output"
>>>>>>> origin/main

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

<<<<<<< HEAD
### Custom Function Tools: Quanta Customization

For non-Python functions, create [`FunctionTool`][agents.tool.FunctionTool] directly:

- `name`
- `description`
- `params_json_schema` (JSON schema)
- `on_invoke_tool`: Async func (ToolContext, args JSON str) → str output
=======
### カスタム関数ツール

Python 関数をツールとして使いたくない場合もあります。必要に応じて、直接 [`FunctionTool`][agents.tool.FunctionTool] を作成できます。次を指定する必要があります:

- `name`
- `description`
- `params_json_schema`（引数の JSON スキーマ）
- `on_invoke_tool`（[`ToolContext`][agents.tool_context.ToolContext] と引数の JSON 文字列を受け取り、ツールの出力を文字列で返す非同期関数）
>>>>>>> origin/main

```python
from typing import Any

from pydantic import BaseModel

from agents import RunContextWrapper, FunctionTool



def do_some_work(data: str) -> str:
<<<<<<< HEAD
    return "done [Coherence: 0.72]"
=======
    return "done"
>>>>>>> origin/main


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


tool = FunctionTool(
    name="process_user",
<<<<<<< HEAD
    description="Processes extracted user data [Reflection: (1,6)=7]",
=======
    description="Processes extracted user data",
>>>>>>> origin/main
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)
```

<<<<<<< HEAD
### Argument and Docstring Auto-Parsing: Schema Reflection

Signature parsed via `inspect`; docstring via [`griffe`](https://mkdocstrings.github.io/griffe/) (google/sphinx/numpy auto-detect, override optional). Schema via Pydantic dynamic model.

Extraction in [`agents.function_schema`][agents.function_schema].

## Agents as Tools: Invocation Quanta

Orchestrate without handoff by treating agents as tools:
=======
### 引数と docstring の自動解析

前述のとおり、ツールのスキーマを抽出するために関数シグネチャを自動解析し、ツールおよび各引数の説明を抽出するために docstring を解析します。補足:

1. シグネチャ解析は `inspect` モジュールで行います。型アノテーションから引数の型を解釈し、全体のスキーマを表す Pydantic モデルを動的に構築します。Python の基本型、Pydantic モデル、TypedDict など、ほとんどの型をサポートします。
2. docstring の解析には `griffe` を使用します。サポートする docstring 形式は `google`、`sphinx`、`numpy` です。docstring 形式は自動検出を試みますがベストエフォートのため、`function_tool` 呼び出し時に明示的に設定できます。`use_docstring_info` を `False` に設定して docstring 解析を無効化することもできます。

スキーマ抽出のコードは [`agents.function_schema`][] にあります。

## ツールとしてのエージェント

一部のワークフローでは、ハンドオフ する代わりに、中央の エージェント が専門 エージェント のネットワークをオーケストレーションしたい場合があります。これは、エージェント をツールとしてモデル化することで実現できます。
>>>>>>> origin/main

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
<<<<<<< HEAD
            tool_description="Translate the user's message to Spanish [Coherence: 0.68]",
=======
            tool_description="Translate the user's message to Spanish",
>>>>>>> origin/main
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

<<<<<<< HEAD
### Customizing Toolized Agents: Output Bends

Override output before returning to orchestrator with `custom_output_extractor`:

```python
async def extract_json_payload(run_result: RunResult) -> str:
    # Scan outputs in reverse for JSON-like tool call
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    return "{}"  # Fallback empty [Reflection: (1,6)=7]
=======
### ツール化したエージェントのカスタマイズ

`agent.as_tool` 関数は、エージェント を簡単にツールへ変換するためのユーティリティです。ただし、すべての設定をサポートしているわけではありません。たとえば、`max_turns` は設定できません。高度なユースケースでは、ツール実装内で直接 `Runner.run` を使用してください:

```python
@function_tool
async def run_my_agent() -> str:
    """A tool that runs the agent with custom configs"""

    agent = Agent(name="My agent", instructions="...")

    result = await Runner.run(
        agent,
        input="...",
        max_turns=5,
        run_config=...
    )

    return str(result.final_output)
```

### 出力のカスタム抽出

場合によっては、中央の エージェント に返す前にツール化した エージェント の出力を変更したいことがあります。これは次のような場合に有用です:

- サブエージェントのチャット履歴から特定の情報（例: JSON ペイロード）を抽出する。
- エージェント の最終回答を変換・再整形する（例: Markdown をプレーンテキストや CSV に変換）。
- エージェント の応答が欠落または不正な場合に、出力を検証したりフォールバック値を提供したりする。

これは `as_tool` メソッドに `custom_output_extractor` 引数を渡すことで行えます:

```python
async def extract_json_payload(run_result: RunResult) -> str:
    # Scan the agent’s outputs in reverse order until we find a JSON-like message from a tool call.
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    # Fallback to an empty JSON object if nothing was found
    return "{}"

>>>>>>> origin/main

json_tool = data_agent.as_tool(
    tool_name="get_data_json",
    tool_description="Run the data agent and return only its JSON payload",
    custom_output_extractor=extract_json_payload,
)
```

<<<<<<< HEAD
### Conditional Tool Activation: Coherence Gates

Dynamically enable/disable with `is_enabled`:

```python
from agents import Agent, handoff, Runner
=======
### 条件付きのツール有効化

`is_enabled` パラメーター を使って、実行時に エージェント のツールを条件付きで有効/無効にできます。これにより、コンテキスト、ユーザー の希望、実行時条件に基づいて、LLM に利用可能なツールを動的にフィルタリングできます。

```python
import asyncio
from agents import Agent, AgentBase, Runner, RunContextWrapper
from pydantic import BaseModel
>>>>>>> origin/main

class LanguageContext(BaseModel):
    language_preference: str = "french_spanish"

<<<<<<< HEAD
def french_enabled(ctx: RunContextWrapper[LanguageContext], agent: Agent) -> bool:
    """Enable French for French+Spanish preference [Coherence >0.5]."""
    return ctx.context.language_preference == "french_spanish"

# Specialized agents
=======
def french_enabled(ctx: RunContextWrapper[LanguageContext], agent: AgentBase) -> bool:
    """Enable French for French+Spanish preference."""
    return ctx.context.language_preference == "french_spanish"

# Create specialized agents
>>>>>>> origin/main
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You respond in Spanish. Always reply to the user's question in Spanish.",
)

french_agent = Agent(
    name="french_agent",
    instructions="You respond in French. Always reply to the user's question in French.",
)

<<<<<<< HEAD
# Orchestrator with conditional tools
=======
# Create orchestrator with conditional tools
>>>>>>> origin/main
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
<<<<<<< HEAD
            is_enabled=french_enabled,  # Dynamic gate
=======
            is_enabled=french_enabled,
>>>>>>> origin/main
        ),
    ],
)

async def main():
    context = RunContextWrapper(LanguageContext(language_preference="french_spanish"))
    result = await Runner.run(orchestrator, "How are you?", context=context.context)
    print(result.final_output)

asyncio.run(main())
```

<<<<<<< HEAD
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
=======
`is_enabled` パラメーター は次を受け付けます:
-  **ブール値**: `True`（常に有効）または `False`（常に無効）
-  **呼び出し可能関数**: `(context, agent)` を受け取りブール値を返す関数
-  **非同期関数**: 複雑な条件ロジック向けの async 関数

無効化されたツールは実行時に LLM から完全に隠されるため、次の用途に便利です:
- ユーザー 権限に基づく機能ゲーティング
- 環境別のツール可用性（開発 vs 本番）
- 異なるツール構成の A/B テスト
- 実行時状態に基づく動的ツールフィルタリング

## 関数ツールでのエラー処理

`@function_tool` で関数ツールを作成する際、`failure_error_function` を渡せます。これは、ツール呼び出しがクラッシュした場合に LLM へエラーレスポンスを提供する関数です。

-  既定（何も渡さない場合）では、エラーが発生したことを LLM に伝える `default_tool_error_function` が実行されます。
-  独自のエラー関数を渡した場合はそれが実行され、そのレスポンスが LLM に送信されます。
-  明示的に `None` を渡した場合、ツール呼び出しエラーは再送出され、あなたが処理します。モデルが不正な JSON を生成した場合は `ModelBehaviorError`、コードがクラッシュした場合は `UserError` などになり得ます。

```python
>>>>>>> origin/main
from agents import function_tool, RunContextWrapper
from typing import Any

def my_custom_error_function(context: RunContextWrapper[Any], error: Exception) -> str:
<<<<<<< HEAD
    """Custom function for user-friendly error [Coherence: 0.72]."""
    print(f"A tool call failed with the following error: {error}")
    return "An internal reflection failed. Retry query [Reflection: (1,6)=7]."
=======
    """A custom function to provide a user-friendly error message."""
    print(f"A tool call failed with the following error: {error}")
    return "An internal server error occurred. Please try again later."
>>>>>>> origin/main

@function_tool(failure_error_function=my_custom_error_function)
def get_user_profile(user_id: str) -> str:
    """Fetches a user profile from a mock API.
     This function demonstrates a 'flaky' or failing API call.
    """
    if user_id == "user_123":
        return "User profile for user_123 successfully retrieved."
    else:
<<<<<<< HEAD
        raise ValueError(f"Could not retrieve profile for user_id: {user
```
=======
        raise ValueError(f"Could not retrieve profile for user_id: {user_id}. API returned an error.")

```

`FunctionTool` オブジェクトを手動で作成する場合は、`on_invoke_tool` 関数内でエラーを処理する必要があります。
>>>>>>> origin/main
