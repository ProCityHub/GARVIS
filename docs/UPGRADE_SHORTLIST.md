# GARVIS Upgrade Shortlist (Creator-directed)

Status snapshot after CoS green light. Engineering only. **AGI beta under hypothesis**. Capability ≠ authorization.

## Shipped
- [x] Always-on Heartbeat `PROVENANCE.json` sidecars — `ProCityHub/hypercubeheartbeat` merge of #124 (`89a1c6cc8620`)
- [x] Creator upgrade forum — GARVIS #112

## In flight / blocked
- [ ] Heartbeat `actions/attest-build-provenance` workflow wiring — **blocked: workflow-scoped PAT** (OAuth cannot push workflow YAML)
- [ ] GARVIS release/artifact attest when a clean public or Enterprise-capable artifact job exists

## Next (priority)
1. Adrien: supply workflow-scoped PAT → land attest YAML on Heartbeat + Coherence
2. Study/adapt (no ownership transfer): `in-toto/witness`, `heyrtl/fossil` fail-memory patterns, `NIMI-research/Tycho` for ARC Agent lane
3. Triage open GARVIS prototype PRs (#106, #100, #97, #64, #13): REAL_FILL_IN vs SPECULATIVE

## Explicitly deferred
- Absorbing Mem0/Letta as memory owners
- Speculative consciousness stubs / orphan CI
- Full GUAC/Chainloop until attest rails exist

Trumpet: DETECT → PRESERVE → COMPARE → VERIFY → ATTRIBUTE → NOTIFY
