---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという用語は多義的です。重要になるコンテキストには次の 2 つの大きな種類があります。

1. コードからローカルに利用できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に利用できるコンテキスト: 応答を生成するときに LLM が参照できるデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンは dataclass や Pydantic オブジェクトを使うことです。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出し、ライフサイクルフックなどにはラッパーオブジェクト `RunContextWrapper[T]` が渡されます。ここで `T` はコンテキストオブジェクトの型を表し、`wrapper.context` 経由でアクセスできます。

 **最も重要な** こととして、特定のエージェント実行におけるすべてのエージェント、ツール関数、ライフサイクルなどは、同じタイプのコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のための文脈データ（例: ユーザー名/uid など、ユーザーに関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
-   補助関数

!!! danger "注意"

    コンテキストオブジェクトは LLM に送信されません。これは純粋にローカルなオブジェクトであり、読み書きやメソッド呼び出しが可能です。

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
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取ることがわかります。ツール実装はコンテキストから読み取ります。
3. エージェントにジェネリックの `UserInfo` を指定し、型チェッカーがエラーを検出できるようにします（たとえば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント / LLM のコンテキスト

LLM が呼び出されるとき、LLM が参照できるデータは会話履歴にあるものだけです。したがって、新しいデータを LLM に利用可能にしたい場合は、その履歴に含める形で行う必要があります。これにはいくつかの方法があります。

1. Agent の `instructions` に追加します。これは "system prompt" や "developer message" とも呼ばれます。system prompt は固定文字列でも、コンテキストを受け取って文字列を出力する動的関数でもかまいません。常に有用な情報（例: ユーザー名や現在の日付）に適した一般的な手法です。
2. `Runner.run` 関数を呼び出すときに `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位に配置されるメッセージを用意できます。
3. 関数ツールを通じて公開します。これは  _オンデマンド_  のコンテキストに有用です。つまり、LLM が必要に応じてデータを要求し、ツールを呼び出してそのデータを取得できます。
4. ファイル検索（retrieval）や Web 検索を使用します。これらは、ファイルやデータベース（ファイル検索）、または Web（Web 検索）から関連データを取得できる特殊なツールです。関連する文脈データで応答を「グラウンディング」するのに有用です。