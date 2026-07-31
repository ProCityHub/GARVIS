---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという語は多義的です。ここで扱う主なコンテキストは 2 種類あります。

1. ローカルにコードから利用できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に利用可能なコンテキスト: 応答生成時に LLM が参照できるデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的には dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出しやライフサイクルフックなどには、`RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型を表し、`wrapper.context` からアクセスできます。

最も **重要** なこと: あるエージェント実行に関わるすべてのエージェント、ツール関数、ライフサイクルなどは、同じコンテキストの _型_ を使わなければなりません。

コンテキストは次のような用途に使えます。

-   実行用のコンテキストデータ（例: ユーザー名/uid など、ユーザー に関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
-   ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM には **送信されません**。これは純粋にローカルなオブジェクトであり、読み書きやメソッド呼び出しが可能です。

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

1. これがコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使えます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取り、実装でコンテキストから読み取っています。
3. 型チェッカーでエラーを検出できるように、エージェントにジェネリックの `UserInfo` を指定します（たとえば、異なるコンテキスト型を取るツールを渡そうとした場合）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント / LLM コンテキスト

LLM が呼び出されるとき、LLM が参照できるのは会話履歴からのデータ **のみ** です。したがって、新しいデータを LLM に利用可能にしたい場合は、その履歴で参照できるようにする必要があります。方法はいくつかあります。

1. エージェントの `instructions` に追加します。これは「system prompt（システムプロンプト）」または「developer message」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でも構いません。常に有用な情報（例: ユーザー の名前や現在の日付）に適した手法です。
2. `Runner.run` 関数を呼び出す際の `input` に追加します。これは `instructions` に追加する手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にメッセージを置けます。
3. 関数ツールで公開します。これは _オンデマンド_ のコンテキストに有用です。LLM はデータが必要になったときに、自身でそのツールを呼び出してデータを取得できます。
4. リトリーバルや Web 検索を使います。これらは、ファイルやデータベース（リトリーバル）または Web（Web 検索）から関連データを取得できる特別なツールです。関連するコンテキストデータに基づいて応答を「グラウンディング」するのに有用です。