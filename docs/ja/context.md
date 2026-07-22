---
search:
  exclude: true
---
# コンテキスト管理

コンテキストという語は多義的です。重視すべきコンテキストには主に 2 つのクラスがあります。

1. コードからローカルに利用可能なコンテキスト: これは、ツール関数の実行時や `on_handoff` のようなコールバック、ライフサイクルフックなどで必要になる可能性のあるデータや依存関係です。
2. LLM に利用可能なコンテキスト: これは、応答を生成する際に LLM が目にするデータです。

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティで表現されます。動作の仕組みは次のとおりです。

1. 任意の Python オブジェクトを作成します。一般的なパターンとして、dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種の実行メソッドに渡します（例: `Runner.run(..., **context=whatever**)`）。
3. すべてのツール呼び出し、ライフサイクルフックなどには `RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型を表し、`wrapper.context` でアクセスできます。

**最も重要** な注意点: 特定のエージェント実行における各エージェント、ツール関数、ライフサイクルなどは、同じコンテキストの型を使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行のためのコンテキストデータ（例: ユーザー名/uid などの ユーザー に関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得用オブジェクトなど）
-   ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM に **送信されません**。これは純粋にローカルなオブジェクトであり、読み取り・書き込み・メソッド呼び出しが可能です。

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
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取り、ツールの実装がコンテキストを読み取っています。
3. 型チェッカーがエラーを検出できるよう、エージェントにジェネリックの `UserInfo` を付与しています（例えば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. `run` 関数にコンテキストを渡します。
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント/LLM のコンテキスト

LLM が呼び出されると、LLM が目にできるデータは会話履歴のもの **のみ** です。つまり、新しいデータを LLM に利用可能にしたい場合は、その履歴で参照できるようにする必要があります。これにはいくつかの方法があります。

1. エージェントの `instructions` に追加します。これは「システムプロンプト」または「開発者メッセージ」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でもかまいません。これは常に有用な情報（例: ユーザー の名前や現在の日付）に対して一般的な戦術です。
2. `Runner.run` 関数を呼び出す際に `input` に追加します。これは `instructions` を使う戦術に似ていますが、[chain of command](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にメッセージを置くことができます。
3. 関数ツール を介して公開します。これはオンデマンドのコンテキストに有用で、LLM 側がいつデータを必要とするかを判断し、ツールを呼び出してそのデータを取得できます。
4. 取得（retrieval）または Web 検索 を使用します。これらは、ファイルやデータベースから関連データを取得（retrieval）したり、ウェブから取得（Web 検索）したりできる特別なツールです。これは、応答を関連するコンテキストデータに「根拠付け」するのに有用です。