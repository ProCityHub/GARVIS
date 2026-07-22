# GARVIS Memory Lifecycle

GARVIS now keeps local SQLite memory beside its provider-independent GGUF
runtime. It recalls only relevant, bounded context and labels every memory
with its evidence status.

Model-generated responses are stored with low confidence as
`model_generated_unverified`; retrieval never upgrades them to evidence.

Automatic maintenance may move memory through:

`active -> consolidated -> latent -> residual trace`

A residual trace keeps only minimal destination/tag/keyword metadata. Full
wording is cleared, and traces are never injected into the model prompt.

## Commands

```bash
uv run --no-dev garvis-memory status
uv run --no-dev garvis-memory remember "Use local GGUF" --kind semantic
uv run --no-dev garvis-memory recall "local model"
uv run --no-dev garvis-memory maintain
uv run --no-dev garvis-memory maintain --apply
```

Environment:

```bash
export GARVIS_MEMORY_DB="$HOME/.garvis/memory_lifecycle.db"
export GARVIS_MEMORY_POLICY="$HOME/GARVIS/config/garvis_memory_policy.json"
export GARVIS_MEMORY_ENABLED=1
```
