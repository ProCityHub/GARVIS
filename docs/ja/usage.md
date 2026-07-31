---
search:
  exclude: true
---
# 使用状況

<<<<<<< HEAD
Agents SDK は、すべての実行ごとにトークン使用状況を自動追跡します。実行コンテキストから参照でき、コストの監視、制限の適用、分析の記録に使えます。

## 追跡対象

- **requests**: 実行された LLM API 呼び出しの数
- **input_tokens**: 送信した入力トークンの合計
=======
Agents SDK は各実行ごとにトークン使用状況を自動追跡します。実行コンテキストからアクセスでき、コストの監視、制限の適用、分析の記録に利用できます。

## 追跡対象

- **requests**: 行われた LLM API 呼び出し数
- **input_tokens**: 送信された入力トークンの合計
>>>>>>> origin/main
- **output_tokens**: 受信した出力トークンの合計
- **total_tokens**: 入力 + 出力
- **details**:
  - `input_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens`

## 実行からの使用状況へのアクセス

<<<<<<< HEAD
`Runner.run(...)` の後、`result.context_wrapper.usage` で使用状況にアクセスします。
=======
`Runner.run(...)` の後、`result.context_wrapper.usage` から使用状況にアクセスします。
>>>>>>> origin/main

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

<<<<<<< HEAD
使用状況は、その実行中のすべてのモデル呼び出し（ツール呼び出しやハンドオフを含む）にわたって集計されます。

### LiteLLM モデルでの使用状況の有効化

LiteLLM プロバイダーはデフォルトでは使用状況メトリクスを報告しません。[`LitellmModel`](models/litellm.md) を使用する場合、エージェントに `ModelSettings(include_usage=True)` を渡すと、LiteLLM のレスポンスが `result.context_wrapper.usage` に反映されます。
=======
使用状況は実行中のすべてのモデル呼び出し（ツール呼び出しや ハンドオフ を含む）で集計されます。

### LiteLLM モデルでの使用状況の有効化

LiteLLM プロバイダーはデフォルトでは使用状況メトリクスを報告しません。[`LitellmModel`](models/litellm.md) を使用する場合は、LiteLLM のレスポンスが `result.context_wrapper.usage` に反映されるよう、エージェントに `ModelSettings(include_usage=True)` を渡してください。
>>>>>>> origin/main

```python
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel

agent = Agent(
    name="Assistant",
    model=LitellmModel(model="your/model", api_key="..."),
    model_settings=ModelSettings(include_usage=True),
)

result = await Runner.run(agent, "What's the weather in Tokyo?")
print(result.context_wrapper.usage.total_tokens)
```

## セッションでの使用状況へのアクセス

<<<<<<< HEAD
`Session`（例: `SQLiteSession`）を使用する場合、`Runner.run(...)` への各呼び出しは、その特定の実行の使用状況を返します。セッションはコンテキストのために会話履歴を保持しますが、各実行の使用状況は独立しています。
=======
`Session`（例: `SQLiteSession`）を使用する場合、`Runner.run(...)` の各呼び出しは、その特定の実行に対する使用状況を返します。セッションはコンテキストのために会話履歴を保持しますが、各実行の使用状況は独立しています。
>>>>>>> origin/main

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

<<<<<<< HEAD
セッションは実行間で会話コンテキストを保持しますが、各 `Runner.run()` 呼び出しで返される使用状況メトリクスは、その実行のみを表します。セッションでは、以前のメッセージが各実行の入力として再投入される場合があり、その結果、以降のターンでの入力トークン数に影響します。

## フックでの使用状況の利用

`RunHooks` を使用している場合、各フックに渡される `context` オブジェクトには `usage` が含まれます。これにより、主要なライフサイクル時点で使用状況を記録できます。
=======
セッションは実行間で会話コンテキストを保持しますが、各 `Runner.run()` 呼び出しで返される使用状況メトリクスはその実行のみを表します。セッションでは、前のメッセージが各実行の入力として再投入されることがあり、その結果、後続ターンの入力トークン数に影響します。

## フックでの使用状況の利用

`RunHooks` を使用している場合、各フックに渡される `context` オブジェクトには `usage` が含まれます。これにより、重要なライフサイクルのタイミングで使用状況を記録できます。
>>>>>>> origin/main

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

## API リファレンス

詳細な API ドキュメントは次を参照してください:

-   [`Usage`][agents.usage.Usage] - 使用状況の追跡データ構造
<<<<<<< HEAD
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - 実行コンテキストから使用状況へアクセス
-   [`RunHooks`][agents.run.RunHooks] - 使用状況追跡ライフサイクルへのフック
=======
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - 実行コンテキストから使用状況にアクセス
-   [`RunHooks`][agents.run.RunHooks] - 使用状況トラッキングのライフサイクルにフック
>>>>>>> origin/main
