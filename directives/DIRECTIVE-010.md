# DIRECTIVE-010 — Cognitive Cycle Engine

## Objective
Build `src/garvis/cognitive_cycle.py`: a CycleEngine class that
executes one full cognitive cycle and emits a snapshot dict that
passes `validate_hypercube_snapshot` with all 15 required fields.

## Hard rules
1. Every emitted snapshot MUST pass validate_hypercube_snapshot.
   Import it; do not redefine the field list.
2. `power_request` defaults to {"requested": false}. Any escalation
   sets requested=true and the engine HALTS the cycle pending
   external approval. No auto-approval path may exist.
3. No network calls. No subprocess. No file writes outside a
   caller-provided snapshot directory.
4. Pure stdlib + existing garvis modules only. No new dependencies.
5. Tests in tests/garvis/test_cognitive_cycle.py. Minimum:
   valid snapshot emission, missing-field rejection, power_request
   halt behavior, deterministic cycle_id sequencing.
6. Vocabulary: software tests use PASS/FAIL only. Never
   SUPPORTED/NOT_SUPPORTED (reserved for empirical claims).

## Forbidden files
- src/agents/** (upstream fork code — do not touch)
- src/garvis/facebook_integration.py
- src/garvis/hypercube_snapshot.py (import only, no edits)
- Any CI/workflow files

## Authority
Adrien D. Thomas retains merge authority. Open a pull request.
