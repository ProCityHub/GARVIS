---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント型の AI アプリを構築できるようにします。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) の本番運用対応アップグレードです。Agents SDK には、ごく少数の基本コンポーネントがあります。

- **エージェント**: instructions と tools を備えた LLM
- **ハンドオフ**: 特定のタスクを別のエージェントに委譲できる仕組み
- **ガードレール**: エージェントの入力と出力を検証できる仕組み
- **セッション**: エージェントの実行をまたいで会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を表現でき、急な学習コストなしに実運用アプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** が付属しており、エージェントのフローを可視化・デバッグできるほか、評価やアプリケーション向けのモデルのファインチューニングにも活用できます。

## Agents SDK を使う理由

この SDK は、次の 2 つの設計原則に基づいています。

1. 使う価値があるだけの機能を備えつつ、学習を素早くできるよう基本コンポーネントは少なく。
2. そのままでも十分に動作し、必要に応じて挙動を正確にカスタマイズ可能に。

SDK の主な機能は次のとおりです。

- エージェントループ: ツールの呼び出し、結果の LLM への送信、LLM が完了するまでのループを処理する組み込みのエージェントループ。
- Python ファースト: 新しい抽象化を学ぶのではなく、言語の組み込み機能でエージェントのオーケストレーションや連携を実現。
- ハンドオフ: 複数のエージェント間で協調・委譲するための強力な機能。
- ガードレール: エージェントと並行して入力バリデーションやチェックを実行し、失敗時には早期に打ち切り。
- セッション: エージェントの実行をまたいだ会話履歴の自動管理により、手動での状態管理が不要。
- 関数ツール: 任意の Python 関数をツール化し、スキーマ自動生成と Pydantic ベースの検証を提供。
- トレーシング: ワークフローの可視化・デバッグ・監視ができ、OpenAI の評価、ファインチューニング、蒸留ツール群も活用可能な組み込みトレーシング。

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