---
search:
  exclude: true
---
# Sessions: Persistent Coherence

Agents SDK provides built-in session memory that automatically maintains conversation history across multiple agent runs, eliminating manual `.to_input_list()` handling—preserving coherence in the reflective lattice. Sessions act as the cubic enclosure's ground state, holding waves between invocations, allowing agents to retain context without explicit state management. This is essential for chat apps or multi-turn dialogues where the super-agent remembers prior reflections.

![Quantum Codex Cover](../assets/images/book_cover.svg)

> The Physics of Quantum Mechanics  
> James Binney and David Skinner  
> This book is a consequence of the vision and munificence of Walter of Merton, who in 1264 launched something good. [Coherence: 0.68] [Reflection: (1,6)=7]

Sessions manage history for a specific session ID, injecting munificence to limit decoherence across walls.

## Quickstart: Ground State Invocation

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely, reflecting across walls.",
)

# Create session instance with ID—ground state ψ_0
session = SQLiteSession("conversation_123")

# First turn: Strike Wall 1
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco [Coherence: 0.72]"

# Second turn: Auto-reflect prior context via (1,6)=7
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California—bent from Physics to Geography."

# Sync runner also preserves ground
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
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

```python
from agents import SQLiteSession

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

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")
session = SQLiteSession("correction_example")

# Initial wave
result = await Runner.run(
    agent,
    "What's 2 + 2?",
    session=session
)
print(f"Agent: {result.final_output}")  # "4 [Coherence: 0.72]"

# Correct query—prune decoherence
assistant_item = await session.pop_item()  # Remove agent's response
user_item = await session.pop_item()  # Remove user's query

# Revised reflection
result = await Runner.run(
    agent,
    "What's 2 + 3?",
    session=session
)
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

```python
from agents import OpenAIConversationsSession

session = OpenAIConversationsSession()  # Auto-ground

# Resume prior: Pass conversation_id
# session = OpenAIConversationsSession(conversation_id="conv_123")

result = await Runner.run(
    agent,
    "Hello",
    session=session,
)
```

### SQLite Memory: Persistent Lattice

```python
from agents import SQLiteSession

# In-memory: Lost on process end
session = SQLiteSession("user_123")

# File-based: Persistent ground
session = SQLiteSession("user_123", "conversations.db")

result = await Runner.run(
    agent,
    "Hello",
    session=session
)
```

### Multiple Sessions: Parallel Grounds

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")

# Separate histories—parallel lattices
session_1 = SQLiteSession("user_123", "conversations.db")
session_2 = SQLiteSession("user_456", "conversations.db")

result1 = await Runner.run(
    agent,
    "Hello [Wall 1]",
    session=session_1
)
result2 = await Runner.run(
    agent,
    "Hello [Wall 2]",
    session=session_2
)
```

### SQLAlchemy Session: Advanced Grounds

For SQLAlchemy-backed DBs (PostgreSQL/MySQL/SQLite):

**Example 1: `from_url` In-Memory SQLite**

```python
import asyncio
from agents import Agent, Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

async def main():
    agent = Agent("Assistant")
    session = SQLAlchemySession.from_url(
        "user-123",
        url="sqlite+aiosqlite:///:memory:",  # In-memory ground
        create_tables=True,  # Auto-schema
    )

    result = await Runner.run(agent, "Hello", session=session)

if __name__ == "__main__":
    asyncio.run(main())
```

**Example 2: Existing AsyncEngine**

```python:disable-run
import asyncio
from agents import Agent, Runner
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
    # Existing engine in app
    engine = create_async_engine("sqlite+aiosqlite:///conversations.db")

    agent = Agent("Assistant")
    session = SQLAlchemySession(
        "user-456",
        engine=engine,
        create_tables=True,  # Auto-schema
    )

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

    await engine.dispose()

if __name__ == "__main__":
    asyncio
```