---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK には、ターミナル上でエージェントの動作を手早く対話的にテストできる `run_demo_loop` が用意されています。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` はループでユーザー入力を促し、各ターン間の会話履歴を保持します。既定では、生成と同時にモデル出力をストリーミングします。上の例を実行すると、 run_demo_loop は対話型のチャットセッションを開始します。継続的に入力を求め、各ターン間の会話全体を記憶するため（エージェントが何について話したかを把握できます）、生成と同時にエージェントの応答をリアルタイムで自動ストリーミングします。

このチャットセッションを終了するには、`quit` または `exit` と入力して Enter を押すか、`Ctrl-D` キーボードショートカットを使用します。