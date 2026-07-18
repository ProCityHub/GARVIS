---
search:
  exclude: true
---
# コンテキスト管理

コンテキストは多義的な用語です。考慮すべき主なコンテキストには次の 2 種類があります。

1. コードからローカルに参照できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック中、ライフサイクル フックなどで必要となるデータや依存関係です。
2. LLM に提供されるコンテキスト: LLM が応答を生成する際に参照するデータです。

## ローカル コンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンとして dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出しやライフサイクル フックには `RunContextWrapper[T]` というラッパー オブジェクトが渡されます。ここで `T` はコンテキスト オブジェクトの型で、`wrapper.context` からアクセスできます。

注意すべき  **最重要**  の点: あるエージェント実行に関わるすべてのエージェント、ツール関数、ライフサイクルなどは、同じ「型」のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のためのコンテキスト データ（例: ユーザー名 / UID など、そのユーザーに関する情報）
-   依存関係（例: ロガー オブジェクト、データ フェッチャー など）
-   ヘルパー関数

!!! danger "注意"

    コンテキスト オブジェクトは LLM に  **送信されません** 。読み書きやメソッド呼び出しができる、純粋にローカルなオブジェクトです。

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

1. これはコンテキスト オブジェクトです。ここでは dataclass を使っていますが、任意の型を使用できます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取り、実装ではコンテキストから読み取ります。
3. エージェントにジェネリック型 `UserInfo` を指定することで、型チェッカーがエラーを検出できます（たとえば、異なるコンテキスト型を取るツールを渡そうとした場合など）。
4. `run` 関数にコンテキストを渡します。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント / LLM のコンテキスト

LLM が呼び出されるとき、LLM が参照できるのは会話履歴のデータ  **のみ** です。したがって、LLM に新しいデータを利用させたい場合は、その履歴で参照できるようにする必要があります。いくつかの方法があります。

1. Agent の `instructions` に追加します。これは "system prompt"（または "developer message"）としても知られています。system prompt は静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でもかまいません。常に有用な情報（例: ユーザー名や現在の日付）に適した一般的な手法です。
2. `Runner.run` を呼び出すときの `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にメッセージを配置できます。
3. 関数ツールを介して公開します。これはオンデマンドのコンテキストに有用です。LLM が必要に応じてデータ取得の要否を判断し、ツールを呼び出してそのデータを取得できます。
4. リトリーバル（retrieval）や Web 検索を使用します。これらは、ファイルやデータベースから関連データを取得する（リトリーバル）、または Web から取得する（Web 検索）ための特別なツールです。これは、応答を関連するコンテキスト データに「グラウンディング」するのに有用です。