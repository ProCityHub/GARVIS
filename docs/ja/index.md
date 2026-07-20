---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント的な AI アプリを構築できます。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) を本番運用レベルにアップグレードしたものです。Agents SDK にはごく少数の基本コンポーネントがあります:

- **エージェント**: instructions と tools を備えた LLM
- **ハンドオフ**: 特定のタスクを他のエージェントに委譲できるしくみ
- **ガードレール**: エージェントの入力と出力を検証するしくみ
- **セッション**: エージェント実行間の会話履歴を自動的に維持

Python と組み合わせることで、これらの基本コンポーネントはツールとエージェント間の複雑な関係を十分に表現でき、急な学習曲線なしに実運用レベルのアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** があり、エージェントのフローを可視化したりデバッグできるほか、評価やモデルの微調整まで行えます。

## Agents SDK を使う理由

SDK の設計原則は次の 2 点です。

1. 使う価値がある十分な機能を備えつつ、学習が速いように基本コンポーネントは少数に保つこと。
2. そのままでも高品質に動作し、必要に応じて挙動を細かくカスタマイズできること。

主な機能は次のとおりです。

- エージェントループ: ツールの呼び出し、結果の LLM への送信、LLM が完了するまでのループを内蔵。
- Python ファースト: 新しい抽象化を学ぶのではなく、言語の機能でエージェントのオーケストレーションや連鎖を実現。
- ハンドオフ: 複数のエージェント間での調整・委譲を実現する強力な機能。
- ガードレール: エージェントと並行して入力検証やチェックを実行し、失敗時は早期に打ち切り。
- セッション: エージェント実行間の会話履歴を自動管理し、手動の状態管理を不要に。
- 関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic による検証を提供。
- トレーシング: ワークフローの可視化・デバッグ・監視を内蔵し、OpenAI の評価、微調整、蒸留ツール群も活用可能。

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

(_実行する場合は、`OPENAI_API_KEY` 環境変数を設定してください_)

```bash
export OPENAI_API_KEY=sk-...
```