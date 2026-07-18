---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージにより、エージェント的な AI アプリを構築できるようにします。これはエージェントに関する以前の実験的取り組みである [Swarm](https://github.com/openai/swarm/tree/main) の本番運用向けアップグレードです。Agents SDK には、非常に小さな基本コンポーネントのセットがあります:

- **エージェント**: instructions とツールを備えた LLM
- **ハンドオフ**: エージェントが特定のタスクを他のエージェントに委譲できる機能
- **ガードレール**: エージェントの入力と出力を検証できる機能
- **セッション**: エージェント実行間で会話履歴を自動的に維持する機能

これらの基本コンポーネントは Python と組み合わせることで、ツールとエージェント間の複雑な関係を表現でき、急な学習コストなしに実用的なアプリケーションを構築できます。さらに、SDK には内蔵の **トレーシング** があり、エージェントのフローを可視化・デバッグできるほか、評価の実行、アプリケーション向けのモデルのファインチューニングまで行えます。

## Agents SDK を使う理由

この SDK の設計原則は次の 2 点です。

1. 学ぶ価値がある十分な機能を備えつつ、学習を素早くするために基本コンポーネントは少数に保つこと。
2. すぐに使える優れたデフォルトを提供しつつ、挙動を細部までカスタマイズできること。

主な機能は次のとおりです。

- Agent ループ: ツールの呼び出し、LLM への実行結果の送信、LLM が完了するまでのループを処理する組み込みのエージェント ループ。
- Python ファースト: 新しい抽象を学ぶ必要はなく、組み込みの言語機能でエージェントをオーケストレーションおよび連鎖可能。
- ハンドオフ: 複数のエージェント間での調整や委譲を可能にする強力な機能。
- ガードレール: エージェントと並行して入力の検証やチェックを実行し、失敗時には早期に中断。
- セッション: エージェント実行間の会話履歴を自動管理し、手動での状態管理を不要に。
- 関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic ベースの検証を提供。
- トレーシング: ワークフローの可視化・デバッグ・監視を可能にし、OpenAI の評価、ファインチューニング、蒸留ツール群も活用可能。

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