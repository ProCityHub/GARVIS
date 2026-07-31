---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) は、抽象化を最小限に抑えた軽量で使いやすいパッケージで、エージェント的な AI アプリを構築できるようにします。これは、以前のエージェント向けの実験的フレームワークである [Swarm](https://github.com/openai/swarm/tree/main) の、本番運用に適したアップグレード版です。Agents SDK には、ごく少数の基本コンポーネントがあります。

-   **エージェント**、instructions と tools を備えた LLM
-   **ハンドオフ**、特定のタスクのために他のエージェントへ委譲できる仕組み
-   **ガードレール**、エージェントの入力と出力を検証する仕組み
-   **セッション**、エージェントの実行をまたいで会話履歴を自動的に保持

これらの基本コンポーネントは Python と組み合わせることで、ツールとエージェント間の複雑な関係性を十分に表現でき、学習コストをかけずに実用的なアプリケーションを構築できます。さらに、SDK には組み込みの **トレーシング** が付属しており、エージェントフローの可視化とデバッグ、評価、さらにはアプリケーション向けのモデルのファインチューニングまで行えます。

## Agents SDK を使う理由

この SDK は、次の 2 つの設計原則に基づいています。

1. 使う価値があるだけの機能を備えつつ、基本コンポーネントを少なくして素早く学べること。
2. すぐに使えて優れた体験を提供しつつ、挙動を細かくカスタマイズできること。

主な機能は次のとおりです。

-   エージェントループ: ツールの呼び出し、実行結果の LLM への送信、LLM が完了するまでのループを内蔵で処理。
-   Python ファースト: 新しい抽象概念を学ぶ必要はなく、言語の機能でエージェントのオーケストレーションと連携を記述。
-   ハンドオフ: 複数のエージェント間の調整と委譲を可能にする強力な機能。
-   ガードレール: エージェントと並行して入力検証やチェックを実行し、失敗時は早期に中断。
-   セッション: エージェントの実行をまたいだ会話履歴を自動管理し、手動の状態管理を不要化。
-   関数ツール: どんな Python 関数でもツール化し、自動スキーマ生成と Pydantic ベースの検証を提供。
-   トレーシング: ワークフローの可視化、デバッグ、監視を可能にし、さらに OpenAI の評価、ファインチューニング、蒸留ツール群を活用可能。

## インストール

```bash
pip install openai-agents
```

## Hello world のコード例

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