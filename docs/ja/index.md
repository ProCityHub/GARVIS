---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント型 AI アプリを構築できるようにします。これは、以前のエージェント向け実験的プロジェクトである [Swarm](https://github.com/openai/swarm/tree/main) の本番運用可能なアップグレード版です。Agents SDK はごく少数の基本コンポーネントで構成されます:

-   **エージェント**: instructions と tools を備えた LLM
-   **ハンドオフ**: エージェントが特定のタスクを他のエージェントに委任できる機能
-   **ガードレール**: エージェントの入力および出力を検証できる機能
-   **セッション**: エージェントの実行をまたいで会話履歴を自動的に維持

Python と組み合わせると、これらの基本コンポーネントだけでツールとエージェント間の複雑な関係を表現でき、急な学習曲線なしに実運用レベルのアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** が付属しており、エージェントのフローを可視化してデバッグできるほか、評価の実施や、アプリケーション向けのモデルのファインチューニングまで行えます。

## Agents SDK を使う理由

SDK の設計原則は 2 つあります:

1. 使う価値のある十分な機能を提供しつつ、学習を素早くするために基本コンポーネントは少数に保つこと。
2. すぐに使えて優れた体験を提供しつつ、挙動を細部までカスタマイズできること。

SDK の主な機能は次のとおりです:

-   エージェントループ: ツールの呼び出し、結果を LLM へ送信、LLM が完了するまでのループ処理を行う組み込みのエージェントループ。
-   Python ファースト: 新しい抽象化を学ぶのではなく、言語の組み込み機能を使ってエージェントのオーケストレーションとチェーン化を実現。
-   ハンドオフ: 複数のエージェント間での調整と委任を可能にする強力な機能。
-   ガードレール: エージェントと並行して入力の検証やチェックを実行し、チェックが失敗した場合は早期に中断。
-   セッション: エージェントの実行をまたぐ会話履歴の自動管理により、手動の状態管理を不要化。
-   関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic ベースのバリデーションを提供。
-   トレーシング: ワークフローの可視化、デバッグ、監視を可能にし、OpenAI の評価、ファインチューニング、蒸留ツール群も活用可能な組み込みトレーシング。

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