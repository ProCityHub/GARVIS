---
search:
  exclude: true
---
# コンテキスト管理

<<<<<<< HEAD
コンテキストという用語は多義的です。重要になるコンテキストには次の 2 つの大きな種類があります。

1. コードからローカルに利用できるコンテキスト: ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要になるデータや依存関係です。
2. LLM に利用できるコンテキスト: 応答を生成するときに LLM が参照できるデータです。
=======
コンテキストという用語は多義的です。考慮すべき主なコンテキストには次の 2 種類があります。

1. コードからローカルに利用できるコンテキスト: これは、ツール関数の実行時、`on_handoff` のようなコールバック、ライフサイクルフックなどで必要となるデータや依存関係です。
2. LLM に利用可能なコンテキスト: これは、LLM が応答を生成する際に参照できるデータです。
>>>>>>> origin/main

## ローカルコンテキスト

これは [`RunContextWrapper`][agents.run_context.RunContextWrapper] クラスと、その中の [`context`][agents.run_context.RunContextWrapper.context] プロパティによって表現されます。仕組みは次のとおりです。

<<<<<<< HEAD
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
=======
1. 任意の Python オブジェクトを作成します。一般的なパターンとしては、dataclass や Pydantic オブジェクトを使います。
2. そのオブジェクトを各種の実行メソッド（例: `Runner.run(..., **context=whatever**)`）に渡します。
3. すべてのツール呼び出し、ライフサイクルフックなどには、`RunContextWrapper[T]` というラッパーオブジェクトが渡されます。ここで `T` はコンテキストオブジェクトの型で、`wrapper.context` を介してアクセスできます。

最も重要な点: 特定のエージェント実行において、エージェント、ツール関数、ライフサイクルなどはすべて、同じ種類（type）のコンテキストを使用しなければなりません。

コンテキストは次のような用途に使えます。

-   実行に関連する状況データ（例: ユーザー名/UID など、ユーザーに関する情報）
-   依存関係（例: ロガーオブジェクト、データ取得コンポーネントなど）
-   ヘルパー関数

!!! danger "注意"

    コンテキストオブジェクトは LLM に送信されません。あくまでローカルなオブジェクトであり、読み書きやメソッド呼び出しが可能です。
>>>>>>> origin/main

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

<<<<<<< HEAD
1. これがコンテキストオブジェクトです。ここでは dataclass を使っていますが、任意の型を使えます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取ることがわかります。ツール実装はコンテキストから読み取ります。
3. エージェントにジェネリックの `UserInfo` を指定し、型チェッカーがエラーを検出できるようにします（たとえば、異なるコンテキスト型を受け取るツールを渡そうとした場合など）。
4. コンテキストは `run` 関数に渡されます。
=======
1. これはコンテキストオブジェクトです。ここでは dataclass を使用していますが、任意の型を使用できます。
2. これはツールです。`RunContextWrapper[UserInfo]` を受け取ることがわかります。ツールの実装はコンテキストから読み取ります。
3. 型チェッカーがエラーを検出できるように、エージェントに汎用型 `UserInfo` を指定します（たとえば、異なるコンテキスト型を取るツールを渡そうとした場合）。
4. `run` 関数にコンテキストを渡します。
>>>>>>> origin/main
5. エージェントはツールを正しく呼び出し、年齢を取得します。

## エージェント / LLM コンテキスト

<<<<<<< HEAD
LLM が呼び出されるとき、LLM が参照できるデータは会話履歴にあるものだけです。したがって、新しいデータを LLM に利用可能にしたい場合は、その履歴に含める形で行う必要があります。これにはいくつかの方法があります。

1. Agent の `instructions` に追加します。これは "system prompt" や "developer message" とも呼ばれます。system prompt は固定文字列でも、コンテキストを受け取って文字列を出力する動的関数でもかまいません。常に有用な情報（例: ユーザー名や現在の日付）に適した一般的な手法です。
2. `Runner.run` 関数を呼び出すときに `input` に追加します。これは `instructions` の手法に似ていますが、[指揮系統](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位に配置されるメッセージを用意できます。
3. 関数ツールを通じて公開します。これは  _オンデマンド_  のコンテキストに有用です。つまり、LLM が必要に応じてデータを要求し、ツールを呼び出してそのデータを取得できます。
4. ファイル検索（retrieval）や Web 検索を使用します。これらは、ファイルやデータベース（ファイル検索）、または Web（Web 検索）から関連データを取得できる特殊なツールです。関連する文脈データで応答を「グラウンディング」するのに有用です。
=======
LLM が呼び出されるとき、LLM が参照できるのは会話履歴のデータのみです。したがって、新しいデータを LLM に利用可能にしたい場合は、その履歴で参照可能になるように提供する必要があります。方法はいくつかあります。

1. エージェントの `instructions` に追加します。これは「システムプロンプト」または「developer message」とも呼ばれます。システムプロンプトは静的な文字列でも、コンテキストを受け取って文字列を出力する動的な関数でもかまいません。これは常に有用な情報（例: ユーザー名や現在の日付）に一般的な手法です。
2. `Runner.run` 関数を呼び出す際の `input` に追加します。これは `instructions` の手法に似ていますが、[chain of command](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command) の下位にメッセージを配置できます。
3. 関数ツールとして公開します。これはオンデマンドのコンテキストに有用です。LLM は必要なときにデータを要求し、ツールを呼び出してそのデータを取得できます。
4. リトリーバルや Web 検索を使用します。これらは、ファイルやデータベースから関連データを取得（retrieval）したり、Web から取得（Web 検索）したりできる特別なツールです。これは、応答を関連するコンテキストデータに「グラウンディング」するのに有用です。
>>>>>>> origin/main
