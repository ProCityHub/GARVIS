# GARVIS lattice body

The lattice body is a dry-run of the fleet order of operations on the heartbeat and action cycle GARVIS already has. It proposes. Adrien D. Thomas is the sole final authority, and the body does not merge, delete, close, force-push, or send.

Law 0 is loaded and checked before the heartbeat is consulted. A cycle whose stamp does not check is not INTEGRATED.

## What one cycle does

1. Read Law 0.
2. WAKE on heartbeat `RECEIVE` / action `OBSERVE`, and retrieve at most four `memoryHints`.
3. ORIENT on heartbeat `SEGMENT`: take handoffs addressed to GARVIS and list the rest.
4. AUTHORIZE with creator authority, the THANOS merge gate, and an input guardrail. Council `approved` stays advisory.
5. ACT only through the tool registry.
6. VERIFY with file existence, a substring, or a SHA-256. If the evidence cannot be opened, the result is `cannot determine`.
7. RECORD, LEARN, and HAND OFF into a local journal file.
8. CLOSE CYCLE when the entry count reaches a Fibonacci cadence (`3, 5, 8, …, 144`).
9. SLEEP by writing a local digest.

The canonical score is imported from `hypercube_brain.core.coherence` and is not redefined:

```text
C = O^1 * A^(1/phi) * B^(1/phi^2)
```

The digest reports that value beside the half-exponent control:

```text
C_unif = O * A^(1/2) * B^(1/2)
```

Both are `HYPOTHESIS_UNDER_TEST`. If any axis is below 0.3, `stabilized_C` is 0 and the state is FRAGMENTED. Empty memory hints add nothing to the hint contribution. An Echo link is not stamped by this adapter.

## Tools

The registry is deny-by-default. Read, verify, and draft tools can be registered. Merge, delete, close, force-push, and outbound send are refused. Network research is checked with the existing capability-broker phrase and then stopped; the dry-run makes no network request.

## Harness

`src/garvis/lattice_harness/` writes `lattice-digest.md`, `lattice-digest.json`, and `journal.jsonl` under the cycle output directory. Secrets are redacted before those files are written. "Wants to raise" in the digest means a queued proposal for Adrien D. Thomas. It is not a claim of inner wants, beliefs, or experience.

GitHub posting is a disabled flag. No posting client is implemented, and no workflow runs this cycle.

## Run the tests

```bash
uv run pytest tests/garvis/test_lattice_body.py -q
```
