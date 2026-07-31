# GARVIS Memory Lifecycle Directive v1

**Authority:** Adrien D. Thomas / ProCityHub  
**Repository:** `~/GARVIS`  
**Branch:** `feature/memory-lifecycle-v1`

This directive installs a local, evidence-aware memory lifecycle and then
performs the Git workflow automatically: validation, commit, push, pull
request, CI watch, squash merge, branch deletion, and local `main` update.

## Memory design

- Working, episodic, semantic, procedural, core, and residual-trace memory.
- Retention fades over time and strengthens through repetition/retrieval.
- Evidence status remains separate from salience or emotion.
- Relevant recall is bounded before reaching the local GGUF model.
- Weak old language may be compressed into a small “ghost trace.”
- Trace records are excluded from prompt recall and cannot count as facts.
- Core/protected memories are never automatically traced.
- Automatic maintenance never hard-deletes.
- Explicit forgetting requires `FORGET-<memory_id>`.

This is a functional software analogy informed by psychology. It is not a
biological simulation and does not establish consciousness.

## Build boundaries

The directive stops on any validation or GitHub failure. It never commits
model weights, credentials, `.secrets`, private keys, or local databases.
It never sends messages, trades, transfers money, or publishes content.
