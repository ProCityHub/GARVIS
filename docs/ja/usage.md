---
search:
  exclude: true
---
# 使用状況

OpenAI Agents SDK は、すべての run のトークン使用状況を自動追跡します。run コンテキストから参照でき、コストの監視、上限の適用、分析の記録に使えます。

## 追跡対象

- **requests**: 実行された LLM API 呼び出し数
- **input_tokens**: 送信した入力トークン合計
- **output_tokens**: 受信した出力トークン合計
- **total_tokens**: input + output
- **details**:
  - `input_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens`

## run からの使用状況の取得

`Runner.run(...)` の後、`result.context_wrapper.usage` で使用状況にアクセスします。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

使用状況は、run 中のすべてのモデル呼び出し（ツール呼び出しや ハンドオフ を含む）にわたって集計されます。

### LiteLLM モデルでの使用状況の有効化

LiteLLM プロバイダーは、デフォルトでは使用状況メトリクスを報告しません。[`LitellmModel`](models/litellm.md) を使用する場合、エージェントに `ModelSettings(include_usage=True)` を渡して、LiteLLM のレスポンスが `result.context_wrapper.usage` を埋めるようにします。

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

## セッションでの使用状況の取得

`Session`（例: `SQLiteSession`）を使う場合、`Runner.run(...)` の各呼び出しは、その特定の run に対する使用状況を返します。セッションはコンテキスト用に会話履歴を保持しますが、各 run の使用状況は独立しています。

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

セッションは run 間で会話コンテキストを保持しますが、各 `Runner.run()` 呼び出しで返される使用状況メトリクスは、その実行のみを表します。セッションでは、前のメッセージが各 run の入力として再投入される場合があり、その結果として後続ターンの入力トークン数に影響します。

## フックでの使用状況の活用

`RunHooks` を使用している場合、各フックに渡される `context` オブジェクトに `usage` が含まれます。これにより、重要なライフサイクル時点で使用状況を記録できます。

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

## API リファレンス

詳細な API ドキュメントは以下を参照してください:

-   [`Usage`][agents.usage.Usage] - 使用状況追跡のデータ構造
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - run コンテキストからの使用状況アクセス
-   [`RunHooks`][agents.run.RunHooks] - 使用状況トラッキングのライフサイクルにフックします