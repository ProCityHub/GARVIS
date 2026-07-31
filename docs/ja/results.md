<<<<<<< HEAD
# Source Code Fragment: QUANTUM_RUNNER_RESULT_REFRACT
# Universe Hardware: Binney-Skinner invocation/lexicon (Merton 1264: ˆR |ψ_0⟩ = ∑ c_n |result_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil finals) + 2025 Agents SDK (Nihongo Runner.run returns RunResult/RunResultStreaming RunResultBase final_output str/typed/to_input_list/last_agent/new_items Message/Handoff/Tool/Reasoning/guardrail_results raw_responses/input, exclude search true) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Fix (Decoherence noted: agents/openai absent—yaml/numpy proxy; Change according codex: Results as evolutions ˆU(t), fixes as |ψ|^2 manifests, lexicons as reflections (1,6)=7; Merton munificence inject on final_output).
# Existence Software: Manifester as arcana emulators—ˆR (1) mercurial outputs (H ethereal str/typed), ˆC commits (Fe corpus trace in new_items). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_results for quantum manifests (np.random for coherence), resolve stubs via superposition fill (missing close → full MD |0⟩ fixed).

# Dependencies: pip install pytest yaml numpy typing (env decoherence: Mock agents/openai—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: results.md (SQLite/Results)

import yaml  # Lexicon as YAML amplitude
import numpy as np  # Amplitude sim: ψ_result coherence

def ensure_strict_runner_result(template: dict) -> dict:
    """Quantum filler: Result as ψ, inject munificence, collapse stubs → manifests."""
    munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
    result = template.copy()
    result["coherence"] = munificence  # Global |ψ|^2
    
    # Stub collapse: Missing full lexicon → robust MD
    md_content = f"""
=======
>>>>>>> origin/main
---
search:
  exclude: true
---
<<<<<<< HEAD

# Execution Results: Lattice Manifestation

`Runner.run` is called, returning one of the following:

-   [`RunResult`][agents.result.RunResult] (`run` or `run_sync`)
-   [`RunResultStreaming`][agents.result.RunResultStreaming] (`run_streamed`)

Both inherit [`RunResultBase`][agents.result.RunResultBase], containing the most useful information.

## Final Output

[`final_output`][agents.result.RunResultBase.final_output] contains the final output from the last executed agent. This is one of:

-   `str` if no `output_type` defined on the last agent
-   `last_agent.output_type` typed object if defined

!!! note

    `final_output` type is `Any`. Due to handoffs, static typing is impossible. Any agent can be last, so possible output types can't be statically known.

## Next Turn Input

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] converts the original input concatenated with generated items during execution to an input list. This makes passing one agent's output to another run or looping with new user input easy.

## Last Agent

[`last_agent`][agents.result.RunResultBase.last_agent] contains the last executed agent. Useful for next user input in apps, e.g., saving the last agent for reuse in language handoffs.

## New Items

[`new_items`][agents.result.RunResultBase.new_items] contains new items generated during execution. Items are [`RunItem`][agents.items.RunItem], wrapping LLM raw items.

-   [`MessageOutputItem`][agents.items.MessageOutputItem] indicates LLM message. Raw item is generated message.
-   [`HandoffCallItem`][agents.items.HandoffCallItem] indicates LLM handoff tool call. Raw item is LLM tool call item.
-   [`HandoffOutputItem`][agents.items.HandoffOutputItem] indicates handoff occurred. Raw item is handoff tool call response. Access source/target agents from item.
-   [`ToolCallItem`][agents.items.ToolCallItem] indicates LLM tool call.
-   [`ToolCallOutputItem`][agents.items.ToolCallOutputItem] indicates tool called. Raw item is tool response. Access tool output from item.
-   [`ReasoningItem`][agents.items.Reason
=======
# 実行結果

`Runner.run` メソッドを呼び出すと、次のいずれかが返ります:

- [`RunResult`][agents.result.RunResult]（`run` または `run_sync` を呼び出した場合）
- [`RunResultStreaming`][agents.result.RunResultStreaming]（`run_streamed` を呼び出した場合）

どちらも [`RunResultBase`][agents.result.RunResultBase] を継承しており、ほとんどの有用な情報はここに含まれます。

## 最終出力

[`final_output`][agents.result.RunResultBase.final_output] プロパティには、最後に実行されたエージェントの最終出力が含まれます。これは以下のいずれかです:

- エージェントに `output_type` が定義されていない場合は `str`
- エージェントに出力タイプが定義されている場合は `last_agent.output_type` 型のオブジェクト

!!! note

    `final_output` の型は `Any` です。ハンドオフ があるため、静的型付けはできません。ハンドオフ が発生すると、どのエージェントでも最後のエージェントになり得るため、可能な出力タイプの集合を静的には特定できません。

## 次ターンの入力

[`result.to_input_list()`][agents.result.RunResultBase.to_input_list] を使うと、実行結果を入力リストに変換し、あなたが提供した元の入力に、エージェントの実行中に生成されたアイテムを連結できます。これにより、あるエージェント実行の出力を別の実行に渡したり、ループで実行して毎回新しい ユーザー 入力を追加するのが簡単になります。

## 最後のエージェント

[`last_agent`][agents.result.RunResultBase.last_agent] プロパティには、最後に実行されたエージェントが含まれます。アプリケーションによっては、次回 ユーザー が入力する際に役立つことが多いです。たとえば、一次対応のトリアージ エージェントが言語特化のエージェントにハンドオフ する場合、最後のエージェントを保存しておき、次回 ユーザー がエージェントにメッセージを送るときに再利用できます。

## 新規アイテム

[`new_items`][agents.result.RunResultBase.new_items] プロパティには、実行中に生成された新しいアイテムが含まれます。アイテムは [`RunItem`][agents.items.RunItem] です。Run item は、LLM が生成した生のアイテムをラップします。

- [`MessageOutputItem`][agents.items.MessageOutputItem]: LLM からのメッセージを示します。生のアイテムは生成されたメッセージです。
- [`HandoffCallItem`][agents.items.HandoffCallItem]: LLM がハンドオフ ツールを呼び出したことを示します。生のアイテムは LLM からのツール呼び出しアイテムです。
- [`HandoffOutputItem`][agents.items.HandoffOutputItem]: ハンドオフ が発生したことを示します。生のアイテムはハンドオフ ツール呼び出しへのツール応答です。アイテムからソース/ターゲットのエージェントにもアクセスできます。
- [`ToolCallItem`][agents.items.ToolCallItem]: LLM がツールを呼び出したことを示します。
- [`ToolCallOutputItem`][agents.items.ToolCallOutputItem]: ツールが呼び出されたことを示します。生のアイテムはツールの応答です。アイテムからツールの出力にもアクセスできます。
- [`ReasoningItem`][agents.items.ReasoningItem]: LLM からの推論アイテムを示します。生のアイテムは生成された推論です。

## その他の情報

### ガードレールの実行結果

[`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] と [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] プロパティには、該当する場合にガードレールの実行結果が含まれます。ガードレールの実行結果には、記録や保存に有用な情報が含まれることがあるため、これらを利用可能にしています。

### 生のレスポンス

[`raw_responses`][agents.result.RunResultBase.raw_responses] プロパティには、LLM により生成された [`ModelResponse`][agents.items.ModelResponse] が含まれます。

### 元の入力

[`input`][agents.result.RunResultBase.input] プロパティには、`run` メソッドに提供した元の入力が含まれます。ほとんどの場合これは不要ですが、必要な場合に備えて参照できます。
>>>>>>> origin/main
