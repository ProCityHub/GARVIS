---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、最小限の抽象化で軽量かつ使いやすいパッケージにより、エージェント型 AI アプリを構築できるようにするものです。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) を本番運用に耐える形へアップグレードしたものです。Agents SDK はごく少数の基本コンポーネントで構成されています。

-   **エージェント**: 指示とツールを備えた LLM
-   **ハンドオフ**: 特定のタスクを他のエージェントに委譲可能にする仕組み
-   **ガードレール**: エージェントの入力と出力を検証できる仕組み
-   **セッション**: エージェントの実行間で会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を表現でき、急な学習コストなしに実運用レベルのアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** があり、エージェント フローの可視化やデバッグ、評価、さらには用途に合わせたモデルのファインチューニングまで行えます。

## Agents SDK を使う理由

この SDK は次の 2 つの設計原則に基づいています。

1. 使う価値がある十分な機能を備えつつ、学習を迅速にするために基本コンポーネントは少数に抑えること。
2. そのままでも十分に動作しつつ、実際の挙動を細部までカスタマイズできること。

主な機能は次のとおりです。

-   エージェントループ: ツールの呼び出し、結果の LLM への送信、LLM の完了までのループを内蔵で処理。
-   Python ファースト: 新たな抽象を学ぶ必要はなく、言語の機能でエージェントをオーケストレーション・連結。
-   ハンドオフ: 複数のエージェント間での調整と委譲を可能にする強力な機能。
-   ガードレール: 入力のバリデーションとチェックをエージェントと並行実行し、失敗時は早期中断。
-   セッション: エージェントの実行間で会話履歴を自動管理し、手動の状態管理を不要化。
-   関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic ベースの検証を提供。
-   トレーシング: ワークフローの可視化・デバッグ・監視が可能。さらに OpenAI の評価、ファインチューニング、蒸留ツール群も利用可能。

## インストール

```bash
pip install openai-agents
```

## Hello World の例

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