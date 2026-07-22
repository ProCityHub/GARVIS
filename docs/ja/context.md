---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという用語は多義的です。ここでは主に次の 2 つのコンテキストを扱います。

1. コードでローカルに利用可能なコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック中、ライフサイクルフックなどで必要となるデータや依存関係です。
2. LLM に利用可能なコンテキスト: 応答を生成する際に LLM が参照できるデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的には dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出しやライフサイクルフックなどには `RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型で、`wrapper.context` からアクセスできます。

**最重要** な注意点: あるエージェント実行において、そのエージェントのすべてのツール関数・ライフサイクルなどは、同じ型のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます:

- 実行のための状況データ（例: ユーザー名/uid など、ユーザーに関する情報）
- 依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
- ヘルパー関数

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

1. これはコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使用できます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取っているのが分かります。ツール実装はコンテキストから読み取ります。
3. 型チェッカーがエラーを検出できるように、エージェントにジェネリクス `UserInfo` を付けます（例えば、異なるコンテキスト型を受け取るツールを渡そうとした場合）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント/LLM コンテキスト

LLM が呼び出されるとき、LLM が参照できるのは会話履歴に含まれるデータのみです。したがって、新しいデータを LLM に利用可能にするには、その履歴に含める形で渡す必要があります。方法は複数あります:

1. Agent の `instructions` に追加します。これは「システムプロンプト」または「開発者メッセージ」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的関数でも構いません。常に有用な情報（例: ユーザーの名前や現在の日付）に適した一般的な方法です。
2. `Runner.run` を呼ぶ際の `input` に追加します。これは `instructions` の方法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command)の下位にメッセージを配置できます。
3. 関数ツールを通じて公開します。これはオンデマンドのコンテキストに有用で、LLM が必要と判断したときにツールを呼び出してデータを取得できます。
4. リトリーバルや Web 検索を使用します。これらは、ファイルやデータベース（リトリーバル）、または Web（Web 検索）から関連データを取得できる特別なツールです。関連する状況データに基づいて応答を「グラウンディング」するのに役立ちます。