# GARVIS Local Language Runtime v1

GARVIS now has a provider-independent local generation path.

- Model weights remain local GGUF files and are ignored by Git.
- Inference uses a locally compiled llama.cpp executable.
- No hosted-model API is called by this runtime.
- Requests receive deterministic filing metadata before generation.
- Outside-world actions remain approval-gated.
- Provisional claims remain provisional rather than becoming facts.

Inspect filing without loading the model:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

Run one local response:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

The existing cloud-backed `garvis` command is not removed in this phase. It can
be migrated only after the local path passes device smoke tests.
