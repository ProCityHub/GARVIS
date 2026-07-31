---
search:
  exclude: true
---
# エージェント

<<<<<<< HEAD
エージェントはアプリの中核となる構成要素です。エージェントは、instructions とツールを設定した大規模言語モデル（ LLM ）です。

## 基本設定

エージェントで最も一般的に設定するプロパティは次のとおりです。

-   `name`: エージェントを識別する必須の文字列です。
-   `instructions`: developer message または システムプロンプト とも呼ばれます。
-   `model`: 使用する LLM と、`model_settings` で temperature、top_p などのモデル調整パラメーターを任意で設定できます。
-   `tools`: エージェントがタスクを達成するために使用できるツールです。
=======
エージェントはアプリの中核となる基本構成要素です。エージェントは、instructions とツールで構成された大規模言語モデル（ LLM ）です。

## 基本設定

一般的に設定するエージェントの主なプロパティは次のとおりです。

- `name`: エージェントを識別する必須の文字列。
- `instructions`: developer メッセージまたは system prompt とも呼ばれます。
- `model`: 使用する LLM と、temperature、top_p などのモデル調整パラメーターを設定する任意の `model_settings`。
- `tools`: エージェントがタスク達成のために使用できるツール。
>>>>>>> origin/main

```python
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="o3-mini",
    tools=[get_weather],
)
```

## コンテキスト

<<<<<<< HEAD
エージェントはその `context` 型に対してジェネリックです。コンテキストは依存性注入のためのツールで、あなたが作成して `Runner.run()` に渡すオブジェクトです。これはすべてのエージェント、ツール、ハンドオフなどに渡され、エージェント実行のための依存関係や状態をまとめて保持します。コンテキストとして任意の Python オブジェクトを提供できます。
=======
エージェントはその `context` 型に対してジェネリックです。コンテキストは依存性注入のためのツールで、あなたが作成して `Runner.run()` に渡すオブジェクトです。これはすべてのエージェント、ツール、ハンドオフなどに渡され、エージェント実行のための依存関係と状態をまとめたものとして機能します。コンテキストには任意の Python オブジェクトを指定できます。
>>>>>>> origin/main

```python
@dataclass
class UserContext:
    name: str
    uid: str
    is_pro_user: bool

    async def fetch_purchases() -> list[Purchase]:
        return ...

agent = Agent[UserContext](
    ...,
)
```

## 出力タイプ

<<<<<<< HEAD
デフォルトでは、エージェントはプレーンテキスト（すなわち `str`）を出力します。特定のタイプの出力を生成したい場合は、`output_type` パラメーターを使用できます。一般的な選択肢としては [Pydantic](https://docs.pydantic.dev/) オブジェクトがありますが、Pydantic の [TypeAdapter](https://docs.pydantic.dev/latest/api/type_adapter/) でラップ可能な任意のタイプ（dataclasses、lists、TypedDict など）をサポートします。
=======
デフォルトでは、エージェントはプレーンテキスト（つまり `str`）を出力します。特定のタイプの出力を生成したい場合は、`output_type` パラメーターを使用できます。一般的な選択肢は [Pydantic](https://docs.pydantic.dev/) オブジェクトの使用ですが、Pydantic の [TypeAdapter](https://docs.pydantic.dev/latest/api/type_adapter/) でラップできる任意の型（dataclasses、リスト、TypedDict など）をサポートします。
>>>>>>> origin/main

```python
from pydantic import BaseModel
from agents import Agent


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
```

!!! note

<<<<<<< HEAD
    `output_type` を渡すと、モデルは通常のプレーンテキストではなく [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) を使用するようになります。

## マルチエージェントのシステム設計パターン

マルチエージェント システムを設計する方法は多数ありますが、一般的に広く適用できる 2 つのパターンをよく見かけます。

1. マネージャー（エージェントをツールとして使用）: 中央のマネージャー/オーケストレーターが、専門化されたサブエージェントをツールとして呼び出し、会話の制御を保持します。
2. ハンドオフ: 対等なエージェント同士で、会話を引き継ぐ専門エージェントに制御を渡します。これは分散型です。

詳しくは [実践的なエージェント構築ガイド](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) を参照してください。

### マネージャー（エージェントをツールとして使用）

`customer_facing_agent` はすべてのユーザー対応を行い、ツールとして公開された専門サブエージェントを呼び出します。詳細は [tools](tools.md#agents-as-tools) ドキュメントをご覧ください。
=======
    `output_type` を指定すると、モデルは通常のプレーンテキスト応答ではなく [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) を使用するようになります。

## マルチエージェントの設計パターン

マルチエージェントシステムの設計方法は数多くありますが、一般的に広く適用できる 2 つのパターンがあります。

1. マネージャー（エージェントをツールとして使用）: 中央のマネージャー／オーケストレーターが、専門化されたサブエージェントをツールとして呼び出し、会話の制御を保持します。
2. ハンドオフ: ピアのエージェントが、会話を引き継ぐ専門エージェントに制御をハンドオフします。これは分散型です。

詳細は [エージェント構築の実践ガイド](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) を参照してください。

### マネージャー（エージェントをツールとして使用）

`customer_facing_agent` はすべてのユーザーとの対話を処理し、ツールとして公開された専門サブエージェントを呼び出します。詳しくは [tools](tools.md#agents-as-tools) ドキュメントをご覧ください。
>>>>>>> origin/main

```python
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

customer_facing_agent = Agent(
    name="Customer-facing agent",
    instructions=(
        "Handle all direct user communication. "
        "Call the relevant tools when specialized expertise is needed."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)
```

### ハンドオフ

<<<<<<< HEAD
ハンドオフは、エージェントが委譲できるサブエージェントです。ハンドオフが発生すると、委譲先のエージェントは会話履歴を受け取り、会話を引き継ぎます。このパターンにより、単一タスクに秀でたモジュール化された専門エージェントが可能になります。詳細は [handoffs](handoffs.md) ドキュメントをご覧ください。
=======
ハンドオフは、エージェントが委任できるサブエージェントです。ハンドオフが発生すると、委任先のエージェントは会話履歴を受け取り、会話を引き継ぎます。このパターンにより、単一のタスクに特化して優れた能力を発揮する、モジュール式の専門エージェントが可能になります。詳しくは [handoffs](handoffs.md) ドキュメントをご覧ください。
>>>>>>> origin/main

```python
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, hand off to the booking agent. "
        "If they ask about refunds, hand off to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
```

## 動的 instructions

<<<<<<< HEAD
多くの場合、エージェントの作成時に instructions を指定できますが、関数を使って動的に instructions を提供することもできます。関数はエージェントとコンテキストを受け取り、プロンプトを返す必要があります。通常の関数と `async` 関数のどちらも受け付けます。
=======
多くの場合、エージェントの作成時に instructions を指定できますが、関数を介して動的な instructions を提供することも可能です。関数はエージェントとコンテキストを受け取り、プロンプトを返す必要があります。通常の関数と `async` 関数のどちらも使用できます。
>>>>>>> origin/main

```python
def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)
```

## ライフサイクルイベント（フック）

<<<<<<< HEAD
場合によっては、エージェントのライフサイクルを観測したくなることがあります。たとえば、イベントをログに記録したり、特定のイベント発生時にデータを事前取得したりする場合です。`hooks` プロパティを使ってエージェントのライフサイクルにフックできます。[`AgentHooks`][agents.lifecycle.AgentHooks] クラスをサブクラス化し、関心のあるメソッドをオーバーライドしてください。

## ガードレール

ガードレールにより、エージェントの実行と並行してユーザー入力に対するチェック/検証を実行し、エージェントの出力が生成された後にもチェック/検証を行えます。たとえば、ユーザー入力とエージェントの出力の関連性をスクリーニングできます。詳細は [guardrails](guardrails.md) ドキュメントをご覧ください。
=======
エージェントのライフサイクルを観察したい場合があります。例えば、イベントをログに記録したり、特定のイベントが発生したときにデータを事前取得したりする場合です。`hooks` プロパティでエージェントのライフサイクルにフックできます。[`AgentHooks`][agents.lifecycle.AgentHooks] クラスを継承し、関心のあるメソッドをオーバーライドしてください。

## ガードレール

ガードレールにより、エージェントの実行と並行してユーザー入力に対するチェック／検証を実行し、エージェントの出力が生成された後にもチェックできます。例えば、ユーザー入力やエージェント出力の関連性をスクリーニングできます。詳しくは [guardrails](guardrails.md) ドキュメントをご覧ください。
>>>>>>> origin/main

## エージェントのクローン/コピー

エージェントの `clone()` メソッドを使用すると、エージェントを複製し、必要に応じて任意のプロパティを変更できます。

```python
pirate_agent = Agent(
    name="Pirate",
    instructions="Write like a pirate",
    model="o3-mini",
)

robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot",
)
```

## ツール使用の強制

<<<<<<< HEAD
ツールのリストを指定しても、必ずしも LLM がツールを使用するとは限りません。[`ModelSettings.tool_choice`][agents.model_settings.ModelSettings.tool_choice] を設定することでツール使用を強制できます。有効な値は次のとおりです。

1. `auto`：LLM がツールを使用するかどうかを判断します。
2. `required`：LLM にツールの使用を要求します（ただし、どのツールを使うかは賢く判断します）。
3. `none`：LLM にツールを使わないことを要求します。
4. 特定の文字列（例：`my_tool`）を設定すると、LLM にその特定のツールの使用を要求します。
=======
ツールのリストを提供しても、LLM が必ずツールを使用するとは限りません。[`ModelSettings.tool_choice`][agents.model_settings.ModelSettings.tool_choice] を設定して、ツール使用を強制できます。有効な値は次のとおりです。

1. `auto`: LLM にツールを使用するかどうかを判断させます。
2. `required`: LLM にツールの使用を要求します（ただし、どのツールを使うかは賢く判断できます）。
3. `none`: LLM にツールを  _使用しない_  よう要求します。
4. 特定の文字列（例: `my_tool`）を設定すると、その特定のツールの使用を LLM に要求します。
>>>>>>> origin/main

```python
from agents import Agent, Runner, function_tool, ModelSettings

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="get_weather")
)
```

## ツール使用時の動作

<<<<<<< HEAD
`Agent` の設定にある `tool_use_behavior` パラメーターは、ツール出力の扱い方を制御します。

- `"run_llm_again"`: デフォルト。ツールを実行し、その結果を LLM が処理して最終応答を生成します。
- `"stop_on_first_tool"`: 最初のツール呼び出しの出力を、その後の LLM 処理なしで最終応答として使用します。
=======
`Agent` 設定の `tool_use_behavior` パラメーターは、ツール出力の扱いを制御します。

- `"run_llm_again"`: デフォルト。ツールを実行し、LLM が結果を処理して最終応答を生成します。
- `"stop_on_first_tool"`: 最初のツール呼び出しの出力を最終応答として使用し、LLM によるその後の処理は行いません。
>>>>>>> origin/main

```python
from agents import Agent, Runner, function_tool, ModelSettings

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    tool_use_behavior="stop_on_first_tool"
)
```

<<<<<<< HEAD
- `StopAtTools(stop_at_tool_names=[...])`: 指定したいずれかのツールが呼び出された場合に停止し、その出力を最終応答として使用します。
=======
- `StopAtTools(stop_at_tool_names=[...])`: 指定したいずれかのツールが呼び出された時点で停止し、その出力を最終応答として使用します。
>>>>>>> origin/main

```python
from agents import Agent, Runner, function_tool
from agents.agent import StopAtTools

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

@function_tool
def sum_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

agent = Agent(
    name="Stop At Stock Agent",
    instructions="Get weather or sum numbers.",
    tools=[get_weather, sum_numbers],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather"])
)
```

- `ToolsToFinalOutputFunction`: ツール結果を処理し、停止するか LLM を継続するかを判断するカスタム関数。

```python
from agents import Agent, Runner, function_tool, FunctionToolResult, RunContextWrapper
from agents.agent import ToolsToFinalOutputResult
from typing import List, Any

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

def custom_tool_handler(
    context: RunContextWrapper[Any],
    tool_results: List[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Processes tool results to decide final output."""
    for result in tool_results:
        if result.output and "sunny" in result.output:
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=f"Final weather: {result.output}"
            )
    return ToolsToFinalOutputResult(
        is_final_output=False,
        final_output=None
    )

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    tool_use_behavior=custom_tool_handler
)
```

!!! note

<<<<<<< HEAD
    無限ループを防ぐため、フレームワークはツール呼び出し後に `tool_choice` を自動的に "auto" にリセットします。この動作は [`agent.reset_tool_choice`][agents.agent.Agent.reset_tool_choice] で設定できます。無限ループが起きるのは、ツール結果が LLM に送られ、`tool_choice` により LLM がさらに別のツール呼び出しを生成し続けてしまうためです。
=======
    無限ループを防ぐため、フレームワークはツール呼び出し後に `tool_choice` を自動的に "auto" にリセットします。この動作は [`agent.reset_tool_choice`][agents.agent.Agent.reset_tool_choice] で設定できます。無限ループは、ツール結果が LLM に送られ、`tool_choice` により LLM が再度ツール呼び出しを生成し続けるために発生します。
>>>>>>> origin/main
