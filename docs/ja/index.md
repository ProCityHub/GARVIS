---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント型の AI アプリを構築できるようにします。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) を本番運用向けに強化したものです。Agents SDK はごく少数の基本コンポーネントを備えています:

-   **エージェント**: instructions と tools を備えた LLM
-   **ハンドオフ**: 特定のタスクを別のエージェントに委譲できる機能
-   **ガードレール**: エージェントの入力と出力を検証できる機能
-   **セッション**: エージェントの実行間で会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントだけでツールとエージェント間の複雑な関係を表現でき、急な学習曲線なしに実運用レベルのアプリケーションを構築できます。加えて、SDK には組み込みの **トレーシング** があり、エージェントのフローを可視化・デバッグできるほか、評価を行い、アプリケーション向けにモデルのファインチューニングまで実施できます。

## Agents SDK を使う理由

この SDK は 2 つの設計原則に基づいています:

1. 使う価値がある十分な機能を備えつつ、学習が速いよう基本コンポーネントは少なく。
2. すぐに使えて優れた体験を提供しつつ、挙動は細部までカスタマイズ可能に。

SDK の主な機能は次のとおりです:

-   エージェント・ループ: ツールの呼び出し、結果の LLM への送信、LLM の完了までのループを処理する組み込みのエージェント・ループ。
-   Python ファースト: 新しい抽象化を学ぶ必要はなく、言語の組み込み機能でエージェントのオーケストレーションや連鎖を実現。
-   ハンドオフ: 複数のエージェント間での調整と委譲を可能にする強力な機能。
-   ガードレール: エージェントと並行して入力の検証やチェックを実行し、チェックが失敗した場合は早期に中断。
-   セッション: エージェントの実行間で会話履歴を自動管理し、手動の状態管理を不要に。
-   関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic によるバリデーションを提供。
-   トレーシング: フローの可視化、デバッグ、監視に加え、OpenAI の評価、ファインチューニング、蒸留ツール群を活用可能な組み込みのトレーシング。

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