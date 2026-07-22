---
search:
  exclude: true
---
# REPL ユーティリティ

SDK では、ターミナルでエージェントの挙動を素早く対話的にテストできるように、`run_demo_loop` を提供しています。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` はループでユーザー入力を促し、ターン間で会話履歴を保持します。デフォルトでは、生成されたモデル出力をそのままストリーミングします。上記の例を実行すると、 run_demo_loop は対話型のチャットセッションを開始します。あなたの入力を継続的に求め、ターン間の会話全体の履歴を記憶するため（エージェントが何について話したかを把握できます）、生成と同時にエージェントの応答をリアルタイムで自動ストリーミングします。

このチャットセッションを終了するには、`quit` または `exit` と入力して（そして Enter を押す）か、`Ctrl-D` のキーボードショートカットを使用してください。