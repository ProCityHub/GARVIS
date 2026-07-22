---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK は、ターミナルで直接、エージェントの動作を素早く対話的にテストできる `run_demo_loop` を提供します。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` は、ループでユーザー入力を促し、発話間の会話履歴を保持します。既定では、モデル出力を生成と同時にストリーミングします。上の例を実行すると、 `run_demo_loop` が対話型のチャットセッションを開始します。継続的に入力を求め、発話間の会話全体の履歴を記憶し（エージェントが何を話したかを把握できるように）、生成と同時にエージェントの応答をリアルタイムで自動ストリーミングします。

このチャットセッションを終了するには、`quit` または `exit` と入力して Enter を押すか、`Ctrl-D` のキーボードショートカットを使用します。