---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント的な AI アプリを構築できるようにします。これは、当社が以前にエージェント向けに実験していた [Swarm](https://github.com/openai/swarm/tree/main) の本番運用可能なアップグレード版です。Agents SDK には、ごく少数の基本コンポーネントがあります。

-   **エージェント**、LLM に instructions と tools を備えたもの
-   **ハンドオフ**、特定のタスクを別のエージェントに委譲できるしくみ
-   **ガードレール**、エージェントの入力と出力を検証できるしくみ
-   **セッション**、エージェントの実行をまたいで会話履歴を自動的に保持するしくみ

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を表現でき、学習コストをかけずに実運用レベルのアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** があり、エージェントのフローを可視化してデバッグしたり、評価したり、アプリケーション向けにモデルをファインチューニングすることもできます。

## Agents SDK の利用理由

この SDK は次の 2 つの設計原則に基づいています。

1. 使う価値があるだけの機能を備えつつ、学習が速く済むように基本コンポーネントは少数に。
2. すぐに高い性能で使える一方で、起こる処理を細かくカスタマイズ可能。

SDK の主な機能は次のとおりです。

-   エージェントループ: ツールの呼び出し、結果を LLM に送信、LLM が完了するまでのループを処理する組み込みのエージェントループ。
-   Python ファースト: 新しい抽象を学ぶ必要なく、言語の組み込み機能でエージェントをオーケストレーションおよび連鎖。
-   ハンドオフ: 複数のエージェント間で連携・委譲する強力な機能。
-   ガードレール: エージェントと並行して入力の検証とチェックを実行し、チェックに失敗した場合は早期に打ち切り。
-   セッション: エージェントの実行をまたいだ会話履歴の自動管理により、手動の状態管理が不要。
-   関数ツール: 任意の Python 関数をツール化し、スキーマの自動生成と Pydantic による検証を提供。
-   トレーシング: ワークフローの可視化、デバッグ、監視を可能にし、OpenAI の評価・ファインチューニング・蒸留ツール群も活用可能。

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