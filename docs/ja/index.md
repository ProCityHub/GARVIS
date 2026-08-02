---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント型の AI アプリを構築できます。これは、エージェント向けのこれまでの試行である [Swarm](https://github.com/openai/swarm/tree/main) を本番運用向けにアップグレードしたものです。Agents SDK には、ごく少数の基本コンポーネントがあります。

-   **エージェント**: instructions と tools を備えた LLM
-   **ハンドオフ**: 特定のタスクを他のエージェントに委譲できる仕組み
-   **ガードレール**: エージェントの入力と出力を検証できる仕組み
-   **セッション**: エージェントの実行間で会話履歴を自動的に維持

これらの基本コンポーネントは Python と組み合わせることで、ツールとエージェント間の複雑な関係を表現でき、急な学習コストなしに実運用アプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** が付属しており、エージェントのフローを可視化してデバッグできるほか、評価や、アプリケーション向けのモデルのファインチューニングも可能です。

## Agents SDK を使う理由

この SDK は、次の 2 つの設計原則に基づいています。

1. 使う価値のある十分な機能を備えつつ、学習が速いように基本コンポーネントは少数にする。
2. すぐに使えて高性能でありながら、挙動を細部までカスタマイズできる。

SDK の主な機能は次のとおりです。

-   エージェントループ: ツールの呼び出し、結果の LLM への送信、LLM が完了するまでのループを処理する組み込みループ。
-   Python ファースト: 新しい抽象化を学ぶのではなく、言語の組み込み機能でエージェントをオーケストレーションおよび連結。
-   ハンドオフ: 複数のエージェント間での調整と委譲を可能にする強力な機能。
-   ガードレール: エージェントと並行して入力の検証やチェックを実行し、失敗時は早期に中断。
-   セッション: エージェントの実行間での会話履歴を自動管理し、手動の状態管理を不要に。
-   関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic ベースの検証を提供。
-   トレーシング: ワークフローの可視化、デバッグ、モニタリングに加え、OpenAI の評価、ファインチューニング、蒸留ツールを活用可能な組み込みトレーシング。

## インストール

```bash
pip install openai-agents
```

## Hello world の例

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