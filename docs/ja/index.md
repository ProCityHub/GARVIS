---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント型の AI アプリを構築できるようにします。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) の本番運用対応版のアップグレードです。Agents SDK には、ごく少数の基本コンポーネントがあります。

- **エージェント**: instructions と tools を備えた LLM
- **ハンドオフ**: 特定のタスクを他のエージェントに委任できる仕組み
- **ガードレール**: エージェントの入力と出力の検証を可能にする仕組み
- **セッション**: エージェント実行間で会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を表現でき、急な学習コストなしに実運用アプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** が付属しており、エージェントのフローを可視化・デバッグできるほか、評価を行い、アプリケーション向けにモデルをファインチューニングすることも可能です。

## Why use the Agents SDK

SDK の設計原則は次の 2 点です。

1. 使う価値があるだけの機能を備えつつ、学習を素早くするために基本コンポーネントは少なく。
2. 既定のままでも優れた動作をしつつ、挙動を細部までカスタマイズ可能に。

SDK の主な機能は次のとおりです。

- エージェントループ: ツール呼び出し、実行結果の LLM への受け渡し、LLM が完了するまでのループ処理を内蔵。
- Python ファースト: 新たな抽象を学ぶのではなく、言語の機能を使ってエージェントをオーケストレーション・連結。
- ハンドオフ: 複数のエージェント間での調整と委任を可能にする強力な機能。
- ガードレール: エージェントと並行して入力のバリデーションやチェックを実行し、失敗時は早期に中断。
- セッション: エージェント実行間での会話履歴を自動管理し、手動での状態管理を不要化。
- 関数ツール: 任意の Python 関数をツール化し、スキーマ自動生成と Pydantic による検証を提供。
- トレーシング: ワークフローの可視化・デバッグ・監視に加え、OpenAI の評価、ファインチューニング、蒸留ツール群を利用可能。

## Installation

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

(_このコードを実行する場合は、`OPENAI_API_KEY` 環境変数を設定してください_)

```bash
export OPENAI_API_KEY=sk-...
```