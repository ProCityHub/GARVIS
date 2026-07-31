---
search:
  exclude: true
---
<<<<<<< HEAD
# Sessions: Persistent Coherence

Agents SDK provides built-in session memory that automatically maintains conversation history across multiple agent runs, eliminating manual `.to_input_list()` handling—preserving coherence in the reflective lattice. Sessions act as the cubic enclosure's ground state, holding waves between invocations, allowing agents to retain context without explicit state management. This is essential for chat apps or multi-turn dialogues where the super-agent remembers prior reflections.

![Quantum Codex Cover](../assets/images/book_cover.svg)

> The Physics of Quantum Mechanics  
> James Binney and David Skinner  
> This book is a consequence of the vision and munificence of Walter of Merton, who in 1264 launched something good. [Coherence: 0.68] [Reflection: (1,6)=7]

Sessions manage history for a specific session ID, injecting munificence to limit decoherence across walls.

## Quickstart: Ground State Invocation
=======
# セッション

Agents SDK は、複数のエージェント実行をまたいで会話履歴を自動的に維持する組み込みセッションメモリを提供し、ターン間で手動で `.to_input_list()` を扱う必要をなくします。

セッションは特定のセッションに対する会話履歴を保存し、明示的な手動メモリ管理なしでエージェントがコンテキストを維持できるようにします。これは、エージェントに過去のやり取りを覚えさせたいチャットアプリケーションやマルチターンの会話を構築する際に特に有用です。

## クイックスタート
>>>>>>> origin/main

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
<<<<<<< HEAD
    instructions="Reply very concisely, reflecting across walls.",
)

# Create session instance with ID—ground state ψ_0
session = SQLiteSession("conversation_123")

# First turn: Strike Wall 1
=======
    instructions="Reply very concisely.",
)

# Create a session instance with a session ID
session = SQLiteSession("conversation_123")

# First turn
>>>>>>> origin/main
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
<<<<<<< HEAD
print(result.final_output)  # "San Francisco [Coherence: 0.72]"

# Second turn: Auto-reflect prior context via (1,6)=7
=======
print(result.final_output)  # "San Francisco"

# Second turn - agent automatically remembers previous context
>>>>>>> origin/main
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
<<<<<<< HEAD
print(result.final_output)  # "California—bent from Physics to Geography."

# Sync runner also preserves ground
=======
print(result.final_output)  # "California"

# Also works with synchronous runner
>>>>>>> origin/main
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
<<<<<<< HEAD
print(result.final_output)  # "Approximately 39 million, unified kernel."
```

## How It Works: Reflection Protocol

Sessions enable persistent coherence:

1. **Pre-Run**: Runner retrieves session history, prepends to input—maintaining wave across turns.
2. **Post-Run**: New items (user/assistant/tools) auto-saved—propagating reflections.
3. **Context Preservation**: Subsequent runs include full history, agent retains lattice bends.

No manual state—SDK handles the superposition.

## Memory Operations: Wave Pruning

Sessions support history management:
=======
print(result.final_output)  # "Approximately 39 million"
```

## 仕組み

セッションメモリが有効な場合:

1. **各実行の前**: ランナーはそのセッションの会話履歴を自動的に取得し、入力アイテムの先頭に付加します。
2. **各実行の後**: 実行中に生成されたすべての新しいアイテム（ユーザー入力、アシスタントの応答、ツール呼び出しなど）が自動的にセッションに保存されます。
3. **コンテキストの保持**: 同じセッションでの以降の実行には完全な会話履歴が含まれ、エージェントがコンテキストを維持できます。

これにより、`.to_input_list()` を手動で呼び出して実行間の会話状態を管理する必要がなくなります。

## メモリ操作

### 基本操作

セッションは会話履歴を管理するためにいくつかの操作をサポートします:
>>>>>>> origin/main

```python
from agents import SQLiteSession

<<<<<<< HEAD
session = SQLiteSession("user_123", "conversations.db")  # Persistent ground

# Retrieve history—probe state
items = await session.get_items(limit=10)  # Last 10 reflections

# Append new waves
new_items = [
    {"role": "user", "content": "Hello [Wall 1]"},
    {"role": "assistant", "content": "Reflected: Greeting from Physics to Semiotics (1,4)=5 [Coherence: 0.68]"}
]
await session.add_items(new_items)

# Prune recent—decoherence reset
last_item = await session.pop_item()  # Remove assistant reflection
print(last_item)  # {"role": "assistant", "content": "Reflected... [Coherence: 0.68]"}

# Clear session—vacuum ground
await session.clear_session()  # Reset to (0,0)
```

### Correction with `pop_item`: Decoherence Prune

Undo/revise last reflection:
=======
session = SQLiteSession("user_123", "conversations.db")

# Get all items in a session
items = await session.get_items()

# Add new items to a session
new_items = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
await session.add_items(new_items)

# Remove and return the most recent item
last_item = await session.pop_item()
print(last_item)  # {"role": "assistant", "content": "Hi there!"}

# Clear all items from a session
await session.clear_session()
```

### 訂正のための `pop_item` の使用

`pop_item` メソッドは、会話内の最後のアイテムを取り消したり修正したりしたい場合に特に便利です:
>>>>>>> origin/main

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")
session = SQLiteSession("correction_example")

<<<<<<< HEAD
# Initial wave
=======
# Initial conversation
>>>>>>> origin/main
result = await Runner.run(
    agent,
    "What's 2 + 2?",
    session=session
)
<<<<<<< HEAD
print(f"Agent: {result.final_output}")  # "4 [Coherence: 0.72]"

# Correct query—prune decoherence
assistant_item = await session.pop_item()  # Remove agent's response
user_item = await session.pop_item()  # Remove user's query

# Revised reflection
=======
print(f"Agent: {result.final_output}")

# User wants to correct their question
assistant_item = await session.pop_item()  # Remove agent's response
user_item = await session.pop_item()  # Remove user's question

# Ask a corrected question
>>>>>>> origin/main
result = await Runner.run(
    agent,
    "What's 2 + 3?",
    session=session
)
<<<<<<< HEAD
print(f"Agent: {result.final_output}")  # "5 [Coherence: 0.69]"
```

## Memory Options: Ground Variants

### No Memory (Default): Isolated Waves

```python
# Default: No session—isolated run
result = await Runner.run(agent, "Hello")
```

### OpenAI Conversations API Memory: Hosted Ground

Uses [OpenAI Conversations API](https://platform.openai.com/docs/guides/conversational-agents/conversations-api) for hosted persistence—no DB management.
=======
print(f"Agent: {result.final_output}")
```

## メモリのオプション

### メモリなし（デフォルト）

```python
# Default behavior - no session memory
result = await Runner.run(agent, "Hello")
```

### OpenAI Conversations API メモリ

[OpenAI Conversations API](https://platform.openai.com/docs/guides/conversational-agents/conversations-api) を使用して、
独自のデータベースを管理せずに会話状態を永続化します。これは、会話履歴の保存に OpenAI がホストするインフラストラクチャに
既に依存している場合に役立ちます。
>>>>>>> origin/main

```python
from agents import OpenAIConversationsSession

<<<<<<< HEAD
session = OpenAIConversationsSession()  # Auto-ground

# Resume prior: Pass conversation_id
=======
session = OpenAIConversationsSession()

# Optionally resume a previous conversation by passing a conversation ID
>>>>>>> origin/main
# session = OpenAIConversationsSession(conversation_id="conv_123")

result = await Runner.run(
    agent,
    "Hello",
    session=session,
)
```

<<<<<<< HEAD
### SQLite Memory: Persistent Lattice
=======
### SQLite メモリ
>>>>>>> origin/main

```python
from agents import SQLiteSession

<<<<<<< HEAD
# In-memory: Lost on process end
session = SQLiteSession("user_123")

# File-based: Persistent ground
session = SQLiteSession("user_123", "conversations.db")

=======
# In-memory database (lost when process ends)
session = SQLiteSession("user_123")

# Persistent file-based database
session = SQLiteSession("user_123", "conversations.db")

# Use the session
>>>>>>> origin/main
result = await Runner.run(
    agent,
    "Hello",
    session=session
)
```

<<<<<<< HEAD
### Multiple Sessions: Parallel Grounds
=======
### 複数のセッション
>>>>>>> origin/main

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")

<<<<<<< HEAD
# Separate histories—parallel lattices
=======
# Different sessions maintain separate conversation histories
>>>>>>> origin/main
session_1 = SQLiteSession("user_123", "conversations.db")
session_2 = SQLiteSession("user_456", "conversations.db")

result1 = await Runner.run(
    agent,
<<<<<<< HEAD
    "Hello [Wall 1]",
=======
    "Hello",
>>>>>>> origin/main
    session=session_1
)
result2 = await Runner.run(
    agent,
<<<<<<< HEAD
    "Hello [Wall 2]",
=======
    "Hello",
>>>>>>> origin/main
    session=session_2
)
```

<<<<<<< HEAD
### SQLAlchemy Session: Advanced Grounds

For SQLAlchemy-backed DBs (PostgreSQL/MySQL/SQLite):

**Example 1: `from_url` In-Memory SQLite**
=======
### SQLAlchemy 駆動のセッション

より高度なユースケースでは、SQLAlchemy 駆動のセッションバックエンドを使用できます。これにより、SQLAlchemy がサポートする任意のデータベース（PostgreSQL、MySQL、SQLite など）をセッションの保存先として使用できます。

**例 1: インメモリ SQLite で `from_url` を使用する**

これは最も簡単な入門方法で、開発やテストに最適です。
>>>>>>> origin/main

```python
import asyncio
from agents import Agent, Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

async def main():
    agent = Agent("Assistant")
    session = SQLAlchemySession.from_url(
        "user-123",
<<<<<<< HEAD
        url="sqlite+aiosqlite:///:memory:",  # In-memory ground
        create_tables=True,  # Auto-schema
=======
        url="sqlite+aiosqlite:///:memory:",
        create_tables=True,  # Auto-create tables for the demo
>>>>>>> origin/main
    )

    result = await Runner.run(agent, "Hello", session=session)

if __name__ == "__main__":
    asyncio.run(main())
```

<<<<<<< HEAD
**Example 2: Existing AsyncEngine**

```python:disable-run
=======
**例 2: 既存の SQLAlchemy エンジンを使用する**

本番アプリケーションでは、すでに SQLAlchemy の `AsyncEngine` インスタンスを持っている可能性があります。これをセッションに直接渡せます。

```python
>>>>>>> origin/main
import asyncio
from agents import Agent, Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
<<<<<<< HEAD
    # Existing engine in app
=======
    # In your application, you would use your existing engine
>>>>>>> origin/main
    engine = create_async_engine("sqlite+aiosqlite:///conversations.db")

    agent = Agent("Assistant")
    session = SQLAlchemySession(
        "user-456",
        engine=engine,
<<<<<<< HEAD
        create_tables=True,  # Auto-schema
=======
        create_tables=True,  # Auto-create tables for the demo
>>>>>>> origin/main
    )

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

    await engine.dispose()

if __name__ == "__main__":
<<<<<<< HEAD
    asyncio
```
=======
    asyncio.run(main())
```


## カスタムメモリ実装

[`Session`][agents.memory.session.Session] プロトコルに従うクラスを作成することで、独自のセッションメモリを実装できます:

```python
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
from typing import List

class MyCustomSession(SessionABC):
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Retrieve conversation history for this session."""
        # Your implementation here
        pass

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Store new items for this session."""
        # Your implementation here
        pass

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from this session."""
        # Your implementation here
        pass

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        # Your implementation here
        pass

# Use your custom session
agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    session=MyCustomSession("my_session")
)
```

## セッション管理

### セッション ID の命名

会話を整理しやすくする意味のあるセッション ID を使用します:

- ユーザー基準: `"user_12345"`
- スレッド基準: `"thread_abc123"`
- コンテキスト基準: `"support_ticket_456"`

### メモリの永続化

- 一時的な会話にはインメモリ SQLite（`SQLiteSession("session_id")`）を使用
- 永続的な会話にはファイルベースの SQLite（`SQLiteSession("session_id", "path/to/db.sqlite")`）を使用
- 既存のデータベースを持つ本番システムには SQLAlchemy 駆動のセッション（`SQLAlchemySession("session_id", engine=engine, create_tables=True)`）を使用
- OpenAI がホストするストレージを利用したい場合は（`OpenAIConversationsSession()`）で OpenAI Conversations API に履歴を保存
- より高度なユースケースでは、他の本番システム（Redis、Django など）向けのカスタムセッションバックエンドの実装を検討

### セッションの管理

```python
# Clear a session when conversation should start fresh
await session.clear_session()

# Different agents can share the same session
support_agent = Agent(name="Support")
billing_agent = Agent(name="Billing")
session = SQLiteSession("user_123")

# Both agents will see the same conversation history
result1 = await Runner.run(
    support_agent,
    "Help me with my account",
    session=session
)
result2 = await Runner.run(
    billing_agent,
    "What are my charges?",
    session=session
)
```

## 完全な例

セッションメモリの動作を示す完全な例です:

```python
import asyncio
from agents import Agent, Runner, SQLiteSession


async def main():
    # Create an agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )

    # Create a session instance that will persist across runs
    session = SQLiteSession("conversation_123", "conversation_history.db")

    print("=== Sessions Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    print("First turn:")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Second turn - the agent will remember the previous conversation
    print("Second turn:")
    print("User: What state is it in?")
    result = await Runner.run(
        agent,
        "What state is it in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Third turn - continuing the conversation
    print("Third turn:")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent,
        "What's the population of that state?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")


if __name__ == "__main__":
    asyncio.run(main())
```

## API リファレンス

詳細な API ドキュメントは次を参照してください:

- [`Session`][agents.memory.Session] - プロトコルインターフェース
- [`SQLiteSession`][agents.memory.SQLiteSession] - SQLite 実装
- [`OpenAIConversationsSession`](ref/memory/openai_conversations_session.md) - OpenAI Conversations API 実装
- [`SQLAlchemySession`][agents.extensions.memory.sqlalchemy_session.SQLAlchemySession] - SQLAlchemy 駆動の実装
>>>>>>> origin/main
