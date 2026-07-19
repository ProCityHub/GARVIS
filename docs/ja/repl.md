---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK は、ターミナルで エージェント の動作を手早く対話的にテストできる `run_demo_loop` を提供します。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` は ループ で ユーザー 入力を促し、ターン間で会話履歴を保持します。デフォルトでは、生成と同時にモデル出力を ストリーミング します。上の例を実行すると、run_demo_loop は対話型のチャットセッションを開始します。あなたの入力を継続的に尋ね、ターン間で会話全体の履歴を記憶します（そのため エージェント は何が議論されたかを把握できます）。また、生成されると同時に エージェント の応答を自動的にリアルタイムで ストリーミング します。

このチャットセッションを終了するには、`quit` または `exit` と入力（して Enter を押下）するか、`Ctrl-D` キーボードショートカットを使用します。