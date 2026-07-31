---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK は、ターミナル上でエージェントの挙動を手早く対話的にテストできる `run_demo_loop` を提供します。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` はループでユーザー入力を促し、ターン間で会話履歴を保持します。既定では、モデル出力を生成と同時にストリーミングします。上の例を実行すると、 run_demo_loop は対話型チャットセッションを開始します。継続的に入力を求め、ターン間で会話全体の履歴を記憶するため（エージェント が何を議論したかを把握できます）、生成されると同時にリアルタイムでエージェント の応答を自動的にストリーミングします。

このチャットセッションを終了するには、単に `quit` または `exit` と入力（さらに Enter キーを押す）するか、キーボードショートカットの Ctrl-D を使用します。