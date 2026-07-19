---
search:
  exclude: true
---
# 結果

`Runner.run` メソッドを呼び出すと、次のいずれかが返ります。

- [`RunResult`][agents.result.RunResult]（`run` または `run_sync` を呼び出した場合）
- [`RunResultStreaming`][agents.result.RunResultStreaming]（`run_streamed` を呼び出した場合）

どちらも [`RunResultBase`][agents.result.RunResultBase] を継承しており、有用な情報の多くはここに含まれます。

## 最終出力

[`final_output`][agents.result.RunResultBase.final_output] プロパティには、最後に実行された エージェント の最終出力が含まれます。これは次のいずれかです。

- `str`（最後の エージェント に `output_type` が定義されていない場合）
- `last_agent.output_type` 型のオブジェクト（ エージェント に出力タイプが定義されている場合）

!!! note

    `final_output` の型は `Any` です。handoffs があるため、静的な型付けはできません。handoffs が発生すると、どの Agent でも最後の エージェント になり得るため、可能な出力タイプの集合を静的には知ることができません。

## 次ターンへの入力

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] を使うと、元の入力に、 エージェント 実行中に生成された項目を連結した入力リストへ変換できます。これにより、ある エージェント 実行の出力を別の実行へ渡したり、ループで実行して毎回新しい ユーザー 入力を追加したりするのが容易になります。

## 最後のエージェント

[`last_agent`][agents.result.RunResultBase.last_agent] プロパティには、最後に実行された エージェント が含まれます。アプリケーションによっては、次回 ユーザー が何かを入力する際に有用です。たとえば、一次トリアージ用の エージェント が言語別の エージェント に handoff する構成であれば、最後の エージェント を保存しておき、次回 ユーザー が エージェント にメッセージを送るときに再利用できます。

## 新規アイテム

[`new_items`][agents.result.RunResultBase.new_items] プロパティには、実行中に生成された新しいアイテムが含まれます。アイテムは [`RunItem`][agents.items.RunItem] です。Run item は、LLM が生成した raw なアイテムをラップします。

- [`MessageOutputItem`][agents.items.MessageOutputItem]: LLM からのメッセージを示します。raw アイテムは生成されたメッセージです。
- [`HandoffCallItem`][agents.items.HandoffCallItem]: LLM が handoff ツールを呼び出したことを示します。raw アイテムは LLM のツール呼び出し項目です。
- [`HandoffOutputItem`][agents.items.HandoffOutputItem]: handoff が発生したことを示します。raw アイテムは handoff ツール呼び出しに対するツールの応答です。アイテムからソース/ターゲットの エージェント にもアクセスできます。
- [`ToolCallItem`][agents.items.ToolCallItem]: LLM がツールを呼び出したことを示します。
- [`ToolCallOutputItem`][agents.items.ToolCallOutputItem]: ツールが呼び出されたことを示します。raw アイテムはツールの応答です。アイテムからツール出力にもアクセスできます。
- [`ReasoningItem`][agents.items.ReasoningItem]: LLM からの推論アイテムを示します。raw アイテムは生成された推論です。

## その他の情報

### ガードレール結果

[`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] と [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] プロパティには、ガードレール の結果（存在する場合）が含まれます。ガードレール の結果には、ログ保存したい有用な情報が含まれることがあるため、参照できるように提供しています。

### Raw レスポンス

[`raw_responses`][agents.result.RunResultBase.raw_responses] プロパティには、LLM が生成した [`ModelResponse`][agents.items.ModelResponse] が含まれます。

### 元の入力

[`input`][agents.result.RunResultBase.input] プロパティには、`run` メソッドに渡した元の入力が含まれます。ほとんどの場合は不要ですが、必要に応じて参照できます。