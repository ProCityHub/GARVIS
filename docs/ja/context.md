---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという用語は多義的です。考慮すべきコンテキストには主に 2 つのクラスがあります。

1. ローカルでコードから利用できるコンテキスト: これは、ツール関数の実行時や `on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に提供されるコンテキスト: これは、 LLM が応答を生成する際に目にするデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンとしては、 dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種 run メソッドに渡します（例: `Runner.run(..., **context=whatever**)`）。
3. すべてのツール呼び出しやライフサイクルフックなどには、 `RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型を表し、 `wrapper.context` からアクセスできます。

**最も重要な** 注意点: あるエージェント実行に関わるすべてのエージェント、ツール関数、ライフサイクルなどは、同じ種類（タイプ）のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のための状況依存データ（例: ユーザー名/uid などその ユーザー に関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得モジュールなど）
-   ヘルパー関数

!!! danger "Note"

    コンテキストオブジェクトは LLM には送信されません。これは純粋にローカルなオブジェクトであり、読み書きやメソッド呼び出しが可能です。

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

1. これはコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使用できます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取っているのがわかります。ツール実装はコンテキストから読み取ります。
3. 型チェッカーがエラーを検出できるように、エージェントにジェネリクスの `UserInfo` を付けます（たとえば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント/LLM コンテキスト

LLM が呼び出されると、見えるデータは会話履歴からのものだけです。つまり、新しいデータを LLM に利用可能にしたい場合は、その履歴で利用できる形で提供する必要があります。これにはいくつかの方法があります。

1. エージェントの `instructions` に追加します。これは "system prompt"（または "developer message"）としても知られています。system prompt は静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でも構いません。常に有用な情報（例: ユーザー名や現在の日付）に適した一般的な手法です。
2. `Runner.run` を呼び出すときに `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にあるメッセージを持たせられます。
3. 関数ツール で公開します。これはオンデマンドのコンテキストに有用です。 LLM が必要なデータを判断し、そのデータを取得するためにツールを呼び出せます。
4. リトリーバル や Web 検索 を使用します。これらは、ファイルやデータベースから関連データを取得（リトリーバル）したり、ウェブから取得（Web 検索）したりできる特別なツールです。関連する状況依存データに基づいて応答を「グラウンディング」するのに有用です。