---
search:
  exclude: true
---
# エージェントの実行

エージェントは [`Runner`][agents.run.Runner] クラスで実行できます。オプションは 3 つあります。

1. [`Runner.run()`][agents.run.Runner.run]: 非同期で実行し、[`RunResult`][agents.result.RunResult] を返します。
2. [`Runner.run_sync()`][agents.run.Runner.run_sync]: 同期メソッドで、内部的には `.run()` を実行します。
3. [`Runner.run_streamed()`][agents.run.Runner.run_streamed]: 非同期で実行し、[`RunResultStreaming`][agents.result.RunResultStreaming] を返します。LLM をストリーミングモードで呼び出し、受信したイベントを逐次ストリームします。

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

詳しくは [実行結果ガイド](results.md) をご覧ください。

## エージェントのループ

`Runner` の run メソッドを使うとき、開始エージェントと入力を渡します。入力は文字列（ユーザーメッセージと見なされます）か、OpenAI Responses API のアイテムのリストのいずれかです。

ランナーは次のループを実行します。

1. 現在のエージェントに対して、現在の入力で LLM を呼び出します。
2. LLM が出力を生成します。
    1. LLM が `final_output` を返した場合、ループを終了し、結果を返します。
    2. LLM がハンドオフした場合、現在のエージェントと入力を更新して、ループを再実行します。
    3. LLM がツール呼び出しを生成した場合、それらのツールを実行し、結果を追加して、ループを再実行します。
3. 渡された `max_turns` を超えた場合、[`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded] 例外を送出します。

!!! note

    LLM の出力が「最終出力」と見なされるルールは、所望の型のテキスト出力を生成し、かつツール呼び出しが存在しないことです。

## ストリーミング

ストリーミングを使うと、LLM の実行中にストリーミングイベントも受け取れます。ストリーム完了後、[`RunResultStreaming`][agents.result.RunResultStreaming] には、生成されたすべての新規出力を含む実行の完全な情報が格納されます。ストリーミングイベントは `.stream_events()` を呼び出して取得できます。詳しくは [ストリーミングガイド](streaming.md) をご覧ください。

## 実行設定

`run_config` パラメーターで、エージェント実行のグローバル設定を構成できます。

-   [`model`][agents.run.RunConfig.model]: 各 Agent の `model` に関係なく、使用するグローバルな LLM モデルを設定できます。
-   [`model_provider`][agents.run.RunConfig.model_provider]: モデル名を解決するためのモデルプロバイダー。デフォルトは OpenAI です。
-   [`model_settings`][agents.run.RunConfig.model_settings]: エージェント固有の設定を上書きします。たとえば、グローバルな `temperature` や `top_p` を設定できます。
-   [`input_guardrails`][agents.run.RunConfig.input_guardrails], [`output_guardrails`][agents.run.RunConfig.output_guardrails]: すべての実行に含める入力/出力ガードレールのリスト。
-   [`handoff_input_filter`][agents.run.RunConfig.handoff_input_filter]: ハンドオフに既定のフィルターがない場合に適用する、全ハンドオフ共通の入力フィルター。入力フィルターにより、新しいエージェントへ送る入力を編集できます。詳細は [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] のドキュメントをご覧ください。
-   [`tracing_disabled`][agents.run.RunConfig.tracing_disabled]: 実行全体の [トレーシング](tracing.md) を無効化できます。
-   [`trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data]: トレースに、LLM やツール呼び出しの入出力など、潜在的に機微なデータを含めるかどうかを設定します。
-   [`workflow_name`][agents.run.RunConfig.workflow_name], [`trace_id`][agents.run.RunConfig.trace_id], [`group_id`][agents.run.RunConfig.group_id]: 実行のトレーシング用ワークフロー名、トレース ID、トレースグループ ID を設定します。少なくとも `workflow_name` の設定を推奨します。グループ ID は任意で、複数実行にまたがるトレースを関連付けできます。
-   [`trace_metadata`][agents.run.RunConfig.trace_metadata]: すべてのトレースに含めるメタデータ。

## 会話／チャットスレッド

いずれかの run メソッドを呼ぶと、1 つ以上のエージェント（ひいては 1 回以上の LLM 呼び出し）が実行される可能性がありますが、これはチャット会話における 1 回の論理ターンを表します。例:

1. ユーザーのターン: ユーザーがテキストを入力
2. Runner の実行: 1 番目のエージェントが LLM を呼び出し、ツールを実行し、2 番目のエージェントへハンドオフ。2 番目のエージェントがさらにツールを実行し、出力を生成。

エージェントの実行終了時に、ユーザーへ何を表示するかを選べます。たとえば、エージェントが生成したすべての新規アイテムを表示する、または最終出力のみを表示する、などです。いずれにせよ、ユーザーが追質問をするかもしれません。その場合は、再度 run メソッドを呼び出せます。

### 手動の会話管理

次のターンの入力を取得するために、[`RunResultBase.to_input_list()`][agents.result.RunResultBase.to_input_list] メソッドを使って、会話履歴を手動管理できます。

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

### Sessions による自動会話管理

より簡単な方法として、[Sessions](sessions.md) を使うと、`.to_input_list()` を手動で呼ばずに会話履歴を自動処理できます。

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

Sessions は自動で次を行います。

-   各実行前に会話履歴を取得
-   各実行後に新規メッセージを保存
-   セッション ID ごとに独立した会話を維持

詳しくは [Sessions のドキュメント](sessions.md) をご覧ください。

## 長時間実行エージェントと human-in-the-loop

Agents SDK の [Temporal](https://temporal.io/) 連携を使うと、human-in-the-loop タスクを含む、耐久性のある長時間実行ワークフローを実行できます。長時間実行タスクを完了させるために Temporal と Agents SDK が連携して動作するデモは[この動画](https://www.youtube.com/watch?v=fFBZqzT4DD8)で、ドキュメントは[こちら](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)をご覧ください。

## 例外

この SDK は特定の状況で例外を送出します。完全な一覧は [`agents.exceptions`][] にあります。概要は次のとおりです。

-   [`AgentsException`][agents.exceptions.AgentsException]: SDK 内で送出されるすべての例外の基底クラスです。その他の特定の例外はここから派生します。
-   [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded]: エージェントの実行が、`Runner.run`、`Runner.run_sync`、または `Runner.run_streamed` に渡した `max_turns` 制限を超えたときに送出されます。指定された対話ターン数内にタスクを完了できなかったことを示します。
-   [`ModelBehaviorError`][agents.exceptions.ModelBehaviorError]: 基盤となるモデル（ LLM ）が想定外または無効な出力を生成した場合に発生します。たとえば次のようなケースがあります。
    -   不正な JSON: 特定の `output_type` が定義されている場合などに、ツール呼び出しや直接の出力で不正な JSON 構造を返す。
    -   予期しないツール関連の失敗: モデルが期待どおりにツールを使用できない。
-   [`UserError`][agents.exceptions.UserError]: SDK を使用するあなた（この SDK を使ってコードを書く人）が誤りをしたときに送出されます。これは通常、不正なコード実装、無効な構成、あるいは SDK の API の誤用が原因です。
-   [`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered], [`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered]: それぞれ、入力ガードレールまたは出力ガードレールの条件が満たされた際に送出されます。入力ガードレールは処理前に受信メッセージを確認し、出力ガードレールはエージェントの最終応答を配信前に確認します。