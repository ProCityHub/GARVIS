---
search:
  exclude: true
---
# ハンドオフ

ハンドオフを使うと、あるエージェントが別のエージェントにタスクを委譲できます。これは、異なるエージェントがそれぞれの分野に特化している場面で特に有用です。例えば、カスタマーサポートアプリでは、注文状況、返金、FAQ などのタスクを個別に担当するエージェントがいるかもしれません。

ハンドオフは LLM からはツールとして表現されます。たとえば `Refund Agent` にハンドオフする場合、ツール名は `transfer_to_refund_agent` になります。

## ハンドオフの作成

すべてのエージェントには [`handoffs`][agents.agent.Agent.handoffs] パラメーターがあり、`Agent` を直接渡すか、ハンドオフをカスタマイズする `Handoff` オブジェクトを渡せます。

Agents SDK が提供する [`handoff()`][agents.handoffs.handoff] 関数でハンドオフを作成できます。この関数では、ハンドオフ先のエージェントに加え、任意のオーバーライドや入力フィルターを指定できます。

### 基本的な使用方法

以下はシンプルなハンドオフの作り方です。

```python
from agents import Agent, handoff

billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")

# (1)!
triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])
```

1. エージェントを直接使う（`billing_agent` のように）ことも、`handoff()` 関数を使うこともできます。

### `handoff()` 関数によるハンドオフのカスタマイズ

[`handoff()`][agents.handoffs.handoff] 関数では、さまざまなカスタマイズが可能です。

-   `agent`: ハンドオフ先のエージェントです。
-   `tool_name_override`: 既定では `Handoff.default_tool_name()` が使われ、`transfer_to_<agent_name>` に解決されます。これを上書きできます。
-   `tool_description_override`: `Handoff.default_tool_description()` による既定のツール説明を上書きします。
-   `on_handoff`: ハンドオフが呼び出された際に実行されるコールバック関数です。ハンドオフが行われると分かった時点でデータ取得を開始するなどに有用です。この関数はエージェントのコンテキストを受け取り、オプションで LLM が生成した入力も受け取れます。入力データは `input_type` パラメーターで制御します。
-   `input_type`: ハンドオフが想定する入力の型（任意）。
-   `input_filter`: 次のエージェントが受け取る入力をフィルタリングできます。詳しくは下記を参照してください。
-   `is_enabled`: ハンドオフが有効かどうか。真偽値、または真偽値を返す関数を指定でき、実行時に動的に有効/無効を切り替えられます。

```python
from agents import Agent, handoff, RunContextWrapper

def on_handoff(ctx: RunContextWrapper[None]):
    print("Handoff called")

agent = Agent(name="My agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    tool_name_override="custom_handoff_tool",
    tool_description_override="Custom description",
)
```

## ハンドオフ入力

状況によっては、ハンドオフを呼び出す際に LLM からデータを渡してほしいことがあります。例えば「エスカレーションエージェント」へのハンドオフを想定すると、記録のために理由を提供してもらいたいかもしれません。

```python
from pydantic import BaseModel

from agents import Agent, handoff, RunContextWrapper

class EscalationData(BaseModel):
    reason: str

async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

agent = Agent(name="Escalation agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)
```

## 入力フィルター

ハンドオフが発生すると、新しいエージェントが会話を引き継ぎ、これまでの会話履歴すべてを見ることができます。これを変更したい場合は、[`input_filter`][agents.handoffs.Handoff.input_filter] を設定できます。入力フィルターは、既存の入力を [`HandoffInputData`][agents.handoffs.HandoffInputData] 経由で受け取り、新しい `HandoffInputData` を返す関数です。

一般的なパターン（例えば履歴からすべてのツール呼び出しを削除するなど）は、[`agents.extensions.handoff_filters`][] に実装されています。

```python
from agents import Agent, handoff
from agents.extensions import handoff_filters

agent = Agent(name="FAQ agent")

handoff_obj = handoff(
    agent=agent,
    input_filter=handoff_filters.remove_all_tools, # (1)!
)
```

1. これにより、`FAQ agent` が呼び出されたときに履歴から自動的にすべてのツールが削除されます。

## 推奨プロンプト

LLM がハンドオフを正しく理解できるように、エージェントにハンドオフに関する情報を含めることを推奨します。推奨プレフィックスは [`agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX`][] にあります。あるいは [`agents.extensions.handoff_prompt.prompt_with_handoff_instructions`][] を呼び出して、推奨データを自動的にプロンプトへ追加できます。

```python
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>.""",
)
```