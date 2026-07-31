---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという語は多義的です。考慮すべき主なコンテキストには次の 2 つの種類があります。

1. コードからローカルに利用できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に利用できるコンテキスト: LLM が応答を生成する際に参照できるデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンは dataclass や Pydantic オブジェクトを使うことです。
2. そのオブジェクトを各種の実行メソッドに渡します（例: `Runner.run(..., **context=whatever**)`）。
3. すべてのツール呼び出しやライフサイクルフックなどに、`RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型で、`wrapper.context` からアクセスできます。

 **最も重要** な注意点: あるエージェント実行に関わるすべてのエージェント、ツール関数、ライフサイクルなどは、同じコンテキストの「型」を使用しなければなりません。

コンテキストは次のような用途に使えます。

- 実行時の状況依存データ（例: ユーザー名/uid やユーザーに関するその他の情報）
- 依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
- ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM に送信されません。あくまでローカルなオブジェクトであり、読み書きやメソッド呼び出しが可能なだけです。

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

1. これはコンテキストオブジェクトです。ここでは dataclass を使用していますが、任意の型を使えます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取ることが分かります。ツール実装はコンテキストから読み取ります。
3. エージェントにジェネリック型 `UserInfo` を付けることで、型チェッカーがエラーを検出できます（例えば、異なるコンテキスト型を取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡されます。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント/LLM のコンテキスト

LLM が呼び出されるとき、参照できるのは会話履歴のデータのみです。つまり、新しいデータを LLM に利用可能にしたい場合は、その履歴で参照可能になる方法で提供する必要があります。方法はいくつかあります。

1. エージェントの `instructions` に追加します。これは「システムプロンプト」や「開発者メッセージ」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的関数でも構いません。常に有用な情報（例: ユーザーの名前や現在の日付）に適した一般的な手法です。
2. `Runner.run` を呼ぶ際の `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command)の下位に位置するメッセージを持てます。
3. 関数ツールで公開します。これはオンデマンドのコンテキストに有用で、LLM が必要なときにデータ取得のためツールを呼び出せます。
4. リトリーバルや Web 検索を使用します。これらは、ファイルやデータベース（リトリーバル）または Web（Web 検索）から関連データを取得できる特別なツールです。関連するコンテキストデータで応答を「グラウンディング」するのに有用です。