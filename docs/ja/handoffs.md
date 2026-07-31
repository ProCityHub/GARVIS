---
search:
  exclude: true
---
# ハンドオフ

ハンドオフは、ある エージェント が別の エージェント にタスクを委譲できるようにします。これは、異なる エージェント がそれぞれ異なる領域を専門とするシナリオで特に有用です。たとえば、カスタマーサポートアプリでは、注文状況、返金、FAQ などのタスクをそれぞれ担当する エージェント が存在し得ます。

ハンドオフは LLM に対してツールとして表現されます。たとえば `Refund Agent` にハンドオフする場合、ツール名は `transfer_to_refund_agent` になります。

## ハンドオフの作成

すべての エージェント は [`handoffs`][agents.agent.Agent.handoffs] パラメーターを持ち、これは `Agent` を直接渡すか、ハンドオフをカスタマイズする `Handoff` オブジェクトを指定できます。

Agents SDK が提供する [`handoff()`][agents.handoffs.handoff] 関数を使ってハンドオフを作成できます。この関数では、委譲先の エージェント を指定し、任意でオーバーライドや入力フィルターを設定できます。

### 基本的な使い方

シンプルなハンドオフの作り方は次のとおりです。

```python
from agents import Agent, handoff

billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")

# (1)!
triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])
```

1. `billing_agent` のように エージェント を直接使うことも、`handoff()` 関数を使うこともできます。

### `handoff()` 関数によるハンドオフのカスタマイズ

[`handoff()`][agents.handoffs.handoff] 関数では、さまざまなカスタマイズが可能です。

-   `agent`: 委譲先の エージェント です。
-   `tool_name_override`: 既定では `Handoff.default_tool_name()` が使用され、`transfer_to_<agent_name>` に解決されます。これを上書きできます。
-   `tool_description_override`: `Handoff.default_tool_description()` による既定のツール説明を上書きします。
-   `on_handoff`: ハンドオフ呼び出し時に実行されるコールバック関数です。ハンドオフが呼ばれたと分かった時点でデータ取得を開始するなどに役立ちます。この関数はエージェントのコンテキストを受け取り、任意で LLM が生成した入力も受け取れます。入力データは `input_type` パラメーターで制御します。
-   `input_type`: ハンドオフで想定される入力の型（任意）。
-   `input_filter`: 次の エージェント が受け取る入力をフィルタリングします。詳細は以下を参照してください。
-   `is_enabled`: ハンドオフを有効にするかどうか。ブール値またはブール値を返す関数を指定でき、実行時に動的に有効・無効を切り替えられます。

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

## ハンドオフの入力

状況によっては、ハンドオフ呼び出し時に LLM によるデータ提供が必要になることがあります。たとえば、「Escalation エージェント」へのハンドオフを考えてみてください。記録のために理由を提供したい場合があります。

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

ハンドオフが発生すると、新しい エージェント が会話を引き継ぎ、過去の会話履歴全体を参照できるようになります。これを変更したい場合は、[`input_filter`][agents.handoffs.Handoff.input_filter] を設定できます。入力フィルターは、既存の入力を [`HandoffInputData`][agents.handoffs.HandoffInputData] 経由で受け取り、新しい `HandoffInputData` を返す関数です。

よくあるパターン（たとえば履歴からすべてのツール呼び出しを削除するなど）は、[`agents.extensions.handoff_filters`][] に実装されています。

```python
from agents import Agent, handoff
from agents.extensions import handoff_filters

agent = Agent(name="FAQ agent")

handoff_obj = handoff(
    agent=agent,
    input_filter=handoff_filters.remove_all_tools, # (1)!
)
```

1. これは、`FAQ agent` が呼び出されたときに履歴からツールを自動的にすべて削除します。

## 推奨プロンプト

LLM がハンドオフを正しく理解できるように、エージェント にハンドオフに関する情報を含めることを推奨します。[`agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX`][] の推奨プレフィックスを使用するか、[`agents.extensions.handoff_prompt.prompt_with_handoff_instructions`][] を呼び出して、推奨データをプロンプトに自動的に追加できます。

```python
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>.""",
)
```