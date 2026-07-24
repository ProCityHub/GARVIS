---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという語には複数の意味があります。ここで重要なのは次の 2 つのクラスです。

1. ローカルにコードから利用できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバックやライフサイクルフックで必要になるデータや依存関係です。
2. LLM に利用可能なコンテキスト: 応答生成時に LLM が参照できるデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的には dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種の実行メソッドに渡します（例: `Runner.run(..., **context=whatever**)`）。
3. すべてのツール呼び出しやライフサイクルフックなどには、`RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はあなたのコンテキストオブジェクトの型で、`wrapper.context` からアクセスできます。

ここで **最も重要** な点: 特定のエージェント実行においては、そのエージェント、ツール関数、ライフサイクルなどが、同じ型のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のための文脈データ（例: ユーザー名や uid、その他の ユーザー に関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得ロジックなど）
-   ヘルパー関数

!!! danger "Note"

    コンテキストオブジェクトは ** LLM に送信されません **。これは純粋にローカルなオブジェクトであり、読み書きしたり、メソッドを呼び出したりできます。

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
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取り、実装はコンテキストから読み取ります。
3. エージェントにジェネリクス `UserInfo` を付け、型チェッカーがエラーを検出できるようにします（たとえば、異なるコンテキスト型を受け取るツールを渡そうとした場合）。
4. `run` 関数にコンテキストを渡します。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント／ LLM コンテキスト

LLM が呼び出されるとき、LLM が参照できるデータは会話履歴に含まれるもの **のみ** です。したがって、新しいデータを LLM に提供したい場合は、その履歴で参照できる形で提供する必要があります。方法はいくつかあります。

1. エージェントの `instructions` に追加します。これは "system prompt" や「開発者メッセージ」とも呼ばれます。system prompt は静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でもかまいません。これは常に有用な情報（例: ユーザー名や現在の日付）に適した一般的な手法です。
2. `Runner.run` 関数を呼び出すときの `input` に追加します。これは `instructions` の手法に似ていますが、[chain of command](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位に位置するメッセージを用意できます。
3. 関数ツール 経由で公開します。これはオンデマンドのコンテキストに有用です。LLM が必要だと判断したときにツールを呼び出してデータを取得できます。
4. 取得（retrieval）や Web 検索 を使用します。これらは、ファイルやデータベースから関連データを取得（retrieval）したり、ウェブから取得したり（Web 検索）できる特別なツールです。これは、応答を関連する文脈データに基づいて根拠づけ（grounding）るのに役立ちます。