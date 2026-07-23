---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK は `run_demo_loop` を提供しており、ターミナル上でエージェントの動作を素早く対話的にテストできます。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` はループでユーザー入力を促し、ターン間で会話履歴を保持します。デフォルトでは、生成と同時にモデルの出力をストリーミングします。上の例を実行すると、 run_demo_loop が対話型のチャットセッションを開始します。継続的に入力を求め、ターン間で会話全体の履歴を記憶するため（エージェントが何について話したかを把握できます）、エージェントの応答を生成しながらリアルタイムに自動でストリーミングします。

このチャットセッションを終了するには、`quit` または `exit` と入力して（ Enter を押す）か、`Ctrl-D` キーボードショートカットを使用します。