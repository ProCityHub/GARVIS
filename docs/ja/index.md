---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージにより、エージェント型の AI アプリを構築できるようにします。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) の実運用対応アップグレードです。Agents SDK には、ごく少数の基本コンポーネントがあります。

-   **エージェント**：instructions と ツール を備えた LLM
-   **ハンドオフ**：特定のタスクを他のエージェントに委譲できる仕組み
-   **ガードレール**：エージェントの入力と出力を検証できる仕組み
-   **セッション**：エージェントの実行間で会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を表現するのに十分強力で、学習コストをかけずに実運用レベルのアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** があり、エージェントフローの可視化やデバッグ、評価に加え、アプリケーション向けにモデルをファインチューニングすることもできます。

## Agents SDK を使う理由

SDK の設計原則は次の 2 つです。

1. 使う価値があるだけの機能を備えつつ、学習を素早くできるよう基本コンポーネントは少なく。
2. そのままでも優れた動作をしつつ、挙動を正確にカスタマイズ可能に。

SDK の主な機能は次のとおりです。

-   Agent loop: ツールの呼び出し、結果の LLM への送信、LLM が完了するまでのループを処理する組み込みのエージェントループ。
-   Python ファースト: 新しい抽象化を学ぶのではなく、言語の組み込み機能でエージェントのオーケストレーションや連携を実現。
-   ハンドオフ: 複数のエージェント間での調整と委譲を可能にする強力な機能。
-   ガードレール: エージェントと並行して入力の検証やチェックを実行し、失敗時は早期に中断。
-   セッション: エージェント実行間の会話履歴を自動管理し、手動の状態管理を不要に。
-   関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic ベースの検証を提供。
-   トレーシング: ワークフローの可視化・デバッグ・監視に加え、OpenAI の評価、ファインチューニング、蒸留ツール群を活用できる組み込みのトレーシング。

## インストール

```bash
pip install openai-agents
```

## Hello World のコード例

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_これを実行する場合は、`OPENAI_API_KEY` 環境変数を設定してください_)

```bash
export OPENAI_API_KEY=sk-...
```