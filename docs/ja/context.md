---
search:
  exclude: true
---
# コンテキスト管理

コンテキストは多義的な用語です。ここでは主に次の 2 つの種類のコンテキストがあります。

1. コードからローカルに利用できるコンテキスト: これは、ツール関数の実行時や `on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に提供されるコンテキスト: これは、応答を生成する際に LLM が参照できるデータです。

## ローカルコンテキスト

これは、[`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。動作の流れは次のとおりです。

1. 任意の Python オブジェクトを作成します。よくあるパターンは dataclass や Pydantic オブジェクトを使うことです。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出しやライフサイクルフックなどには、`RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型を表し、`wrapper.context` からアクセスできます。

**最も重要** な点: あるエージェント実行におけるすべてのエージェント、ツール関数、ライフサイクルなどは、同じ種類（同一の型）のコンテキストを使わなければなりません。

コンテキストは次のような用途に使えます。

-   実行のためのコンテキストデータ（例: ユーザー名 / uid などの ユーザー 情報）
-   依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
-   ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM に送信されません。これは純粋にローカルなオブジェクトであり、読み書きやメソッド呼び出しができます。

```python
import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

@dataclass
class UserInfo:  # (1)!
    name: str
    uid: int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:  # (2)!
    """Fetch the age of the user. Call this function to get user's age information."""
    return f"The user {wrapper.context.name} is 47 years old"

async def main():
    user_info = UserInfo(name="John", uid=123)

    agent = Agent[UserInfo](  # (3)!
        name="Assistant",
        tools=[fetch_user_age],
    )

    result = await Runner.run(  # (4)!
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )

    print(result.final_output)  # (5)!
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main())
```

1. これはコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使えます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取っているのが分かります。ツールの実装はコンテキストから読み取ります。
3. 型チェッカーがエラーを検出できるよう、エージェントに総称型の `UserInfo` を指定します（例えば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡します。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント / LLM のコンテキスト

LLM が呼び出されるとき、LLM が参照できるのは会話履歴に含まれるデータのみです。したがって、新しいデータを LLM に利用可能にしたい場合は、そのデータを会話履歴で参照できる形で提供する必要があります。方法はいくつかあります。

1. エージェントの `instructions` に追加します。これは "system prompt"（または「開発者メッセージ」）とも呼ばれます。system prompt は静的な文字列でも、コンテキストを受け取って文字列を出力する動的関数でもかまいません。常に有用な情報（例: ユーザーの名前や現在の日付）に適した一般的な手法です。
2. `Runner.run` を呼び出す際の `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にあるメッセージを使える点が異なります。
3. 関数ツール を通じて公開します。これはオンデマンドのコンテキストに有用です。LLM が必要なときにデータを要求し、ツールを呼び出して取得できます。
4. リトリーバル または Web 検索 を使用します。これらは、ファイルやデータベース（リトリーバル）または Web（Web 検索）から関連データを取得できる特別なツールです。これは、関連するコンテキストデータに基づいて応答を「グラウンディング」するのに有用です。