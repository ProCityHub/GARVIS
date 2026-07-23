---
search:
  exclude: true
---
# コンテキスト管理

コンテキストは多義的な用語です。考慮すべきコンテキストには主に 2 つのクラスがあります。

1. コードでローカルに利用可能なコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に利用可能なコンテキスト: 応答を生成する際に LLM が見るデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンは dataclass や Pydantic オブジェクトを使用することです。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出しやライフサイクルフックにはラッパーオブジェクト `RunContextWrapper[T]` が渡されます。ここで `T` はコンテキストオブジェクトの型を表し、`wrapper.context` でアクセスできます。

 **最も重要** な点: あるエージェント実行におけるすべてのエージェント、ツール関数、ライフサイクルなどは、同じ種類（_type_）のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のための状況データ（例: ユーザー名/uid など、そのユーザー に関する情報）
-   依存関係（例: ロガーオブジェクト、データフェッチャーなど）
-   ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM には送信されません。これは純粋にローカルなオブジェクトであり、読み書きやメソッド呼び出しができます。

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

1. これがコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使用できます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取り、実装ではコンテキストから読み取っています。
3. 型チェッカーでエラーを検出できるよう、エージェントにジェネリック `UserInfo` を指定します（たとえば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント/LLM のコンテキスト

LLM が呼び出されると、LLM が見られるデータは会話履歴にあるものだけです。したがって、新しいデータを LLM で利用可能にしたい場合は、その履歴で利用可能になるような方法で追加する必要があります。いくつかの方法があります。

1. エージェントの `instructions` に追加します。これは「システムプロンプト」または「開発者メッセージ」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でもかまいません。常に有用な情報（例: ユーザー の名前や現在の日付）には一般的な手法です。
2. `Runner.run` 関数を呼び出すときに `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にあるメッセージを用意できます。
3. 関数ツールで公開します。これは _オンデマンド_ コンテキストに有用です。LLM が必要なときを判断し、ツールを呼び出してデータを取得できます。
4. リトリーバルや Web 検索を使用します。これらは、ファイルやデータベースから関連データを取得（リトリーバル）したり、Web（Web 検索）から取得できる特別なツールです。これは、関連する文脈データに基づかせる（グラウンディング）ために有用です。