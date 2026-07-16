---
search:
  exclude: true
---
# 結果

`Runner.run` メソッドを呼び出すと、次のいずれかを取得します。

- [`RunResult`][agents.result.RunResult]（`run` または `run_sync` を呼んだ場合）
- [`RunResultStreaming`][agents.result.RunResultStreaming]（`run_streamed` を呼んだ場合）

どちらも [`RunResultBase`][agents.result.RunResultBase] を継承しており、有用な情報の大半はここに含まれます。

## 最終出力

[`final_output`][agents.result.RunResultBase.final_output] プロパティには、最後に実行されたエージェントの最終出力が含まれます。これは次のいずれかです。

- エージェントで `output_type` が定義されていない場合は `str`
- エージェントで出力タイプが定義されている場合は `last_agent.output_type` 型のオブジェクト

!!! note

    `final_output` の型は `Any` です。ハンドオフのため、静的に型付けできません。ハンドオフが発生する可能性があると、どのエージェントが最後になるか不定で、可能な出力タイプの集合を静的には特定できないためです。

## 次のターンへの入力

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] を使うと、実行結果を、元の入力にエージェント実行中に生成されたアイテムを連結した入力リストへと変換できます。これにより、あるエージェント実行の出力を別の実行に渡したり、ループで実行して毎回新しい ユーザー 入力を追加したりするのが便利になります。

## 最後のエージェント

[`last_agent`][agents.result.RunResultBase.last_agent] プロパティには、最後に実行されたエージェントが含まれます。アプリケーションによっては、次回 ユーザー が入力する際にこれが役立つことが多いです。たとえば、一次トリアージのエージェントが言語別エージェントにハンドオフする構成では、最後のエージェントを保存して、次回 ユーザー がメッセージを送る際に再利用できます。

## 新規アイテム

[`new_items`][agents.result.RunResultBase.new_items] プロパティには、実行中に生成された新しいアイテムが含まれます。アイテムは [`RunItem`][agents.items.RunItem] です。Run item は、LLM が生成した生のアイテムをラップします。

- [`MessageOutputItem`][agents.items.MessageOutputItem] は、LLM からのメッセージを示します。生のアイテムは生成されたメッセージです。
- [`HandoffCallItem`][agents.items.HandoffCallItem] は、LLM がハンドオフ ツールを呼び出したことを示します。生のアイテムは LLM からのツール呼び出しアイテムです。
- [`HandoffOutputItem`][agents.items.HandoffOutputItem] は、ハンドオフが発生したことを示します。生のアイテムはハンドオフ ツール呼び出しに対するツールの応答です。アイテムからソース/ターゲットのエージェントにもアクセスできます。
- [`ToolCallItem`][agents.items.ToolCallItem] は、LLM がツールを呼び出したことを示します。
- [`ToolCallOutputItem`][agents.items.ToolCallOutputItem] は、ツールが呼び出されたことを示します。生のアイテムはツールの応答です。アイテムからツールの出力にもアクセスできます。
- [`ReasoningItem`][agents.items.ReasoningItem] は、LLM からの推論アイテムを示します。生のアイテムは生成された推論です。

## その他の情報

### ガードレール結果

[`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] と [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] プロパティには、（存在する場合）ガードレールの実行結果が含まれます。ガードレールの結果には、ログや保存をしたい有用な情報が含まれることがあるため、これらを参照可能にしています。

### 生の応答

[`raw_responses`][agents.result.RunResultBase.raw_responses] プロパティには、LLM によって生成された [`ModelResponse`][agents.items.ModelResponse] が含まれます。

### 元の入力

[`input`][agents.result.RunResultBase.input] プロパティには、`run` メソッドに渡した元の入力が含まれます。たいていの場合これは不要ですが、必要になったときのために参照できます。