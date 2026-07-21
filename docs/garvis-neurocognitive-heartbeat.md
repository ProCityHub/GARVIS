# GARVIS Neurocognitive Heartbeat

Architect: **Adrien D. Thomas**  
Organization: **ProCityHub**

This runtime separates permanent memory from active working context.

## Heartbeat

- **0.0** stores the exact episode.
- **0.2** segments the signal.
- **0.4** identifies intent and candidate interpretations.
- **0.6** retrieves only relevant memories and checks evidence status.
- **0.8** preserves unresolved questions as speculative dream seeds.
- **1.0** assembles a bounded language-model context.
- **1.2** produces the response.
- **1.4** records execution or prediction errors.
- **1.6** consolidates the turn into episodic and semantic memory.

## Safety and evidence

The engine labels records as verified, supplied, inferred, speculative,
unknown, or retracted. Dream records cannot become verified evidence
without a separate verification process.

The exact episode archive is not loaded into every model request. Selective
recall is used instead, preventing context-window overload while preserving
long-term access to prior records.

## Run

```bash
cd ~/GARVIS
uv run --no-dev python scripts/garvis_neuro_chat.py --session neuro-0.1
```

Status:

```bash
uv run --no-dev python scripts/garvis_neuro_status.py
```
