# GARVIS conversational runtime

The GARVIS response spine restores ordinary question answering while keeping outside-world actions
under explicit human control.

## What changed

The runtime separates two concerns:

- **Conversation:** questions, explanations, analysis, calculations, drafts, plans, summaries, and
  code are answered normally.
- **Execution:** sending, publishing, deleting remote data, changing live accounts, financial
  transactions, and similar side effects require Adrien D Thomas's exact approval immediately
  before execution.

No outside-world tools are attached to the default runtime. The assistant can therefore answer and
prepare work without accidentally sending, deleting, publishing, or transacting.

## Setup

Use Python 3.9 or newer for the current repository codebase. Install the project with `uv`, then set
an API key in the environment. Never commit API keys.

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

The model can be selected with `GARVIS_MODEL`; the default is `gpt-5.6-luna`.

```bash
export GARVIS_MODEL="gpt-5.6-luna"
```

## Run one request

```bash
uv run garvis "Explain the current heartbeat status"
```

## Start an interactive conversation

```bash
uv run garvis --interactive --session adrien
```

Conversation history is stored by default in `~/.garvis/sessions.db`. Use `--no-memory` for an
ephemeral session or `--db PATH` to choose another SQLite database.

## Approval behavior

A request such as `How do I delete an old branch safely?` is informational and receives a normal
answer. A request such as `Delete the remote branch now` is treated as an execution request. GARVIS
may prepare the exact command and explain the consequences, but the external action remains pending
until Adrien gives exact approval and an approved tool is attached.

## Architecture

`garvis.assistant.GarvisAssistant` owns the conversational agent and session memory. The request
assessment is non-blocking metadata; it does not replace or suppress the model response. Action
approval belongs at the tool boundary, not at the question-answering boundary.

Authorship: **Adrien D Thomas / ProCityHub**.
