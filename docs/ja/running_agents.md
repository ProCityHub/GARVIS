---
search:
  exclude: true
---
# エージェントの実行

エージェントは [`Runner`][agents.run.Runner] クラスで実行できます。方法は 3 つあります:

1. [`Runner.run()`][agents.run.Runner.run]: 非同期で実行し、[`RunResult`][agents.result.RunResult] を返します。
2. [`Runner.run_sync()`][agents.run.Runner.run_sync]: 同期メソッドで、内部的には `.run()` を実行します。
3. [`Runner.run_streamed()`][agents.run.Runner.run_streamed]: 非同期で実行し、[`RunResultStreaming`][agents.result.RunResultStreaming] を返します。LLM をストリーミング モードで呼び出し、受信したイベントを順次ストリーミングします。

```python
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance
```

詳細は [結果ガイド](results.md) を参照してください。

## エージェントのループ

`Runner` の run メソッドを使うとき、開始するエージェントと入力を渡します。入力は、文字列（ユーザー メッセージとして扱われます）か、OpenAI Responses API の入力アイテムのリストのいずれかです。

runner は次のループを実行します:

1. 現在のエージェントに対して、現在の入力で LLM を呼び出します。
2. LLM が出力を生成します。
    1. LLM が `final_output` を返した場合、ループを終了し結果を返します。
    2. LLM がハンドオフを行った場合、現在のエージェントと入力を更新し、ループを再実行します。
    3. LLM がツール呼び出しを生成した場合、それらのツール呼び出しを実行し、結果を追記して、ループを再実行します。
3. 渡された `max_turns` を超えた場合、[`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded] 例外を送出します。

!!! note

    LLM の出力が「最終出力」と見なされるルールは、目的の型のテキスト出力を生成し、ツール呼び出しが存在しないことです。

## ストリーミング

ストリーミングでは、LLM の実行中にストリーミング イベントも受け取れます。ストリームが完了すると、[`RunResultStreaming`][agents.result.RunResultStreaming] に、生成されたすべての新しい出力を含む実行に関する完全な情報が含まれます。ストリーミング イベントは `.stream_events()` を呼び出して取得できます。詳細は [ストリーミング ガイド](streaming.md) を参照してください。

## 実行設定

`run_config` パラメーターでは、エージェント実行のグローバル設定を構成できます:

-   [`model`][agents.run.RunConfig.model]: 各 Agent の `model` 設定に関わらず、使用するグローバルな LLM モデルを設定します。
-   [`model_provider`][agents.run.RunConfig.model_provider]: モデル名を解決するためのモデル プロバイダーで、既定は OpenAI です。
-   [`model_settings`][agents.run.RunConfig.model_settings]: エージェント固有の設定を上書きします。たとえば、グローバルな `temperature` や `top_p` を設定できます。
-   [`input_guardrails`][agents.run.RunConfig.input_guardrails], [`output_guardrails`][agents.run.RunConfig.output_guardrails]: すべての実行に含める入力/出力ガードレールのリストです。
-   [`handoff_input_filter`][agents.run.RunConfig.handoff_input_filter]: ハンドオフに既存のフィルターがない場合に適用する、すべてのハンドオフ向けのグローバル入力フィルターです。入力フィルターにより、新しいエージェントへ送信する入力を編集できます。詳細は [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] のドキュメントを参照してください。
-   [`tracing_disabled`][agents.run.RunConfig.tracing_disabled]: 実行全体の [トレーシング](tracing.md) を無効にできます。
-   [`trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data]: LLM やツール呼び出しの入出力など、機微なデータをトレースに含めるかどうかを構成します。
-   [`workflow_name`][agents.run.RunConfig.workflow_name], [`trace_id`][agents.run.RunConfig.trace_id], [`group_id`][agents.run.RunConfig.group_id]: 実行のトレーシング ワークフロー名、トレース ID、トレース グループ ID を設定します。少なくとも `workflow_name` の設定を推奨します。グループ ID は任意で、複数の実行にまたがるトレースをリンクできます。
-   [`trace_metadata`][agents.run.RunConfig.trace_metadata]: すべてのトレースに含めるメタデータです。

## 会話/チャットスレッド

任意の run メソッドの呼び出しは、1 つ以上のエージェント（ひいては 1 回以上の LLM 呼び出し）を実行する場合がありますが、チャット会話における単一の論理ターンを表します。例:

1. ユーザーのターン: ユーザーがテキストを入力
2. Runner の実行: 最初のエージェントが LLM を呼び出し、ツールを実行し、2 番目のエージェントへハンドオフ、2 番目のエージェントがさらにツールを実行し、最終的な出力を生成。

エージェントの実行終了時に、ユーザーへ何を表示するかを選べます。たとえば、エージェントが生成したすべての新しいアイテムを表示するか、最終出力のみを表示します。いずれにせよ、ユーザーが追質問をするかもしれないため、その場合は再度 run メソッドを呼び出せます。

### 手動の会話管理

次のターン用の入力を取得するには、[`RunResultBase.to_input_list()`][agents.result.RunResultBase.to_input_list] メソッドを使って、会話履歴を手動で管理できます:

```python
async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California
```

### Sessions を用いた自動会話管理

より簡単な方法として、[Sessions](sessions.md) を使うと、`.to_input_list()` を手動で呼び出さずに会話履歴を自動で扱えます:

```python
from agents import Agent, Runner, SQLiteSession

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create session instance
    session = SQLiteSession("conversation_123")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
        print(result.final_output)
        # San Francisco

        # Second turn - agent automatically remembers previous context
        result = await Runner.run(agent, "What state is it in?", session=session)
        print(result.final_output)
        # California
```

Sessions は自動的に次を行います:

-   各実行前に会話履歴を取得
-   各実行後に新しいメッセージを保存
-   異なるセッション ID ごとに会話を分離して維持

詳細は [Sessions のドキュメント](sessions.md) を参照してください。

## 長時間実行エージェントと human-in-the-loop（人間参加型）

Agents SDK の [Temporal](https://temporal.io/) 連携を使うと、human-in-the-loop のタスクを含む、永続的で長時間実行のワークフローを実行できます。Temporal と Agents SDK が連携して長時間実行タスクを完了するデモは [この動画](https://www.youtube.com/watch?v=fFBZqzT4DD8) を参照し、ドキュメントは [こちら](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents) を参照してください。

## 例外

SDK は特定のケースで例外を送出します。完全な一覧は [`agents.exceptions`][] にあります。概要は以下のとおりです:

-   [`AgentsException`][agents.exceptions.AgentsException]: SDK 内で送出されるすべての例外の基底クラスです。その他の特定の例外はすべて、ここから派生します。
-   [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded]: エージェントの実行が `Runner.run`、`Runner.run_sync`、または `Runner.run_streamed` メソッドに渡した `max_turns` 制限を超えた場合に送出されます。指定した対話ターン数内にエージェントがタスクを完了できなかったことを示します。
-   [`ModelBehaviorError`][agents.exceptions.ModelBehaviorError]: 基盤モデル（LLM）が予期しない、または無効な出力を生成したときに発生します。これには次が含まれます:
    -   不正な JSON: 特定の `output_type` が定義されている場合に、ツール呼び出しや直接の出力で不正な JSON 構造を返す。
    -   予期しないツール関連の失敗: モデルが期待どおりの方法でツールを使用できない。
-   [`UserError`][agents.exceptions.UserError]: SDK を利用する開発者（あなた）が、SDK の使用中にエラーを起こしたときに送出されます。これは通常、不正なコード実装、無効な構成、または SDK の API の誤用に起因します。
-   [`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered], [`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered]: それぞれ、入力ガードレールまたは出力ガードレールの条件が満たされたときに送出されます。入力ガードレールは処理前に受信メッセージを確認し、出力ガードレールはエージェントの最終応答を配信前に確認します。