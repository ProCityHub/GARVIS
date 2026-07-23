---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えつつ軽量で使いやすいパッケージで、エージェント的な AI アプリを構築できるようにします。これは、以前のエージェント向け実験である [Swarm](https://github.com/openai/swarm/tree/main) の本番運用可能なアップグレードです。Agents SDK には、ごく少数の基本的なコンポーネントがあります。

- **エージェント**: instructions と tools を備えた LLM
- **ハンドオフ**: 特定のタスクで他のエージェントへ委譲できる仕組み
- **ガードレール**: エージェントの入力と出力を検証する仕組み
- **セッション**: エージェントの実行間で会話履歴を自動的に維持

これらのコンポーネントは Python と組み合わせることで、ツールとエージェント間の複雑な関係を表現でき、急な学習曲線なしに実運用アプリケーションを構築できます。さらに、この SDK には組み込みの **トレーシング** があり、エージェントのフローを可視化・デバッグし、評価したり、アプリケーション向けにモデルを微調整することもできます。

## Why use the Agents SDK

この SDK の設計原則は 2 つあります。

1. 使う価値がある十分な機能を備えつつ、学習が速いように基本的なコンポーネントは少数に保つこと。
2. すぐに使える状態で素晴らしく動作しつつ、発生する挙動を正確にカスタマイズできること。

SDK の主な機能は次のとおりです。

- エージェント ループ: ツール呼び出し、結果の LLM への送信、LLM が完了するまでのループを処理する組み込みループ。
- Python ファースト: 新しい抽象を学ぶ必要はなく、言語の組み込み機能でエージェントのオーケストレーションや連鎖を実現。
- ハンドオフ: 複数のエージェント間での協調や委譲を可能にする強力な機能。
- ガードレール: エージェントと並行して入力の検証やチェックを実行し、失敗時は早期に中断。
- セッション: エージェントの実行間で会話履歴を自動管理し、手動での状態管理を不要に。
- 関数ツール: 任意の Python 関数をツール化し、自動スキーマ生成と Pydantic による検証を提供。
- トレーシング: ワークフローの可視化、デバッグ、監視を可能にし、OpenAI の評価・微調整・蒸留ツール群も活用可能。

## インストール

```bash
pip install openai-agents
```

## Hello world 例

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

( _これを実行する場合は、`OPENAI_API_KEY` 環境変数を設定してください_ )

```bash
export OPENAI_API_KEY=sk-...
```