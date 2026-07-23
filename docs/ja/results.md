---
search:
  exclude: true
---
# 結果

`Runner.run` メソッドを呼び出すと、次のいずれかが返ります:

-   [`RunResult`][agents.result.RunResult]（`run` または `run_sync` を呼び出した場合）
-   [`RunResultStreaming`][agents.result.RunResultStreaming]（`run_streamed` を呼び出した場合）

いずれも [`RunResultBase`][agents.result.RunResultBase] を継承しており、主な有用情報はここに含まれます。

## 最終出力

[`final_output`][agents.result.RunResultBase.final_output] プロパティには、最後に実行されたエージェントの最終出力が含まれます。これは次のいずれかです:

-   最後のエージェントに `output_type` が定義されていない場合は `str`
-   エージェントに出力タイプが定義されている場合は `last_agent.output_type` 型のオブジェクト

!!! note

    `final_output` の型は `Any` です。ハンドオフがあるため、静的型付けはできません。ハンドオフが発生する可能性があるということは、どのエージェントでも最後のエージェントになり得るため、可能な出力タイプの集合を静的には特定できないためです。

## 次ターンの入力

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] を使うと、実行時に生成されたアイテムを、元の入力に連結した入力リストへと変換できます。これにより、あるエージェント実行の出力を別の実行へ渡したり、ループで実行して毎回新しい ユーザー 入力を追加したりするのが容易になります。

## 最後のエージェント

[`last_agent`][agents.result.RunResultBase.last_agent] プロパティには、最後に実行されたエージェントが含まれます。アプリケーションによっては、次回 ユーザー が何かを入力する際に有用です。たとえば、フロントラインのトリアージ エージェントが言語別のエージェントへハンドオフする場合、最後のエージェントを保存しておき、次回 ユーザー がエージェントにメッセージを送る際に再利用できます。

## 新規アイテム

[`new_items`][agents.result.RunResultBase.new_items] プロパティには、実行中に生成された新しいアイテムが含まれます。アイテムは [`RunItem`][agents.items.RunItem] です。ランアイテムは、LLM が生成した生のアイテムをラップします。

-   [`MessageOutputItem`][agents.items.MessageOutputItem]: LLM からのメッセージを示します。生のアイテムは生成されたメッセージです。
-   [`HandoffCallItem`][agents.items.HandoffCallItem]: LLM がハンドオフ ツールを呼び出したことを示します。生のアイテムは LLM のツール呼び出しアイテムです。
-   [`HandoffOutputItem`][agents.items.HandoffOutputItem]: ハンドオフが発生したことを示します。生のアイテムはハンドオフ ツール呼び出しに対するツールのレスポンスです。アイテムからソース/ターゲットのエージェントにもアクセスできます。
-   [`ToolCallItem`][agents.items.ToolCallItem]: LLM がツールを呼び出したことを示します。
-   [`ToolCallOutputItem`][agents.items.ToolCallOutputItem]: ツールが呼び出されたことを示します。生のアイテムはツールのレスポンスです。アイテムからツール出力にもアクセスできます。
-   [`ReasoningItem`][agents.items.ReasoningItem]: LLM からの推論アイテムを示します。生のアイテムは生成された推論です。

## その他の情報

### ガードレールの結果

[`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] と [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] プロパティには、該当する場合、ガードレールの結果が含まれます。ガードレールの結果には、ログや保存に有用な情報が含まれることがあるため、利用できるようにしています。

### 生の応答

[`raw_responses`][agents.result.RunResultBase.raw_responses] プロパティには、LLM によって生成された [`ModelResponse`][agents.items.ModelResponse] が含まれます。

### 元の入力

[`input`][agents.result.RunResultBase.input] プロパティには、`run` メソッドに渡した元の入力が含まれます。多くの場合これは不要ですが、必要な際のために利用可能です。