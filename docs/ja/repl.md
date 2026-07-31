---
search:
  exclude: true
---
# REPL ユーティリティ

この SDK には、ターミナル上でエージェントの動作を素早くインタラクティブにテストできる `run_demo_loop` が用意されています。

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())
```

`run_demo_loop` はループでユーザー入力を求め、ターン間で会話履歴を保持します。デフォルトでは、生成と同時にモデルの出力をストリーミングします。上の例を実行すると、run_demo_loop はインタラクティブなチャットセッションを開始します。継続的にあなたの入力を求め、ターン間の会話履歴全体を保持し（そのためエージェントは何が話されたかを把握できます）、生成と同時にエージェントの応答をリアルタイムで自動ストリーミングします。

このチャットセッションを終了するには、`quit` または `exit` と入力して Enter を押すか、`Ctrl-D` のキーボードショートカットを使用してください。
