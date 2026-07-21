# DIRECTIVE-014 — Static ARC Solver (Track A): Audit and Ratified Build Plan

Authority: Adrien D. Thomas, ProCityHub. Addressed to: GARVIS (Claude backend).
Subject: Your document "GARVIS ARC-AGI Solver Framework v1.0".

## Part 1 — Claims audit (respond with corrections acknowledged)

1. "Production-ready framework" and "fully implemented in the provided
   codebase" — NOT_SUPPORTED / RETRACT. At the time you wrote it, no DSL
   primitives, no synthesis engine, no ARCMemory, no puzzle loader existed
   in the repository. The only ARC code was the six DIRECTIVE-011 modules,
   which target interactive ARC-3, not static puzzles. Restate these as
   PROPOSED architecture, not implemented fact.
2. Claims to preserve: the architecture class (DSL program synthesis +
   verification, LLM-guided hypotheses, persistent memory) is the
   historically winning approach for static ARC — external fact. Your
   BoundedSession configuration citations were accurate — implementation
   fact.
3. Benchmark precision: your document targets STATIC grid puzzles
   (ARC-AGI-1/2). Interactive ARC-AGI-3 is a separate track with a separate
   architecture (DIRECTIVE-012). ARC Prize 2026 runs both; Track A targets
   the ARC-AGI-2 grand-prize track.

## Part 2 — Ratified build plan (already in motion)

Your architecture is ACCEPTED as the Track A roadmap under standing rules
(one module per branch, tests at 100 percent, merge only on green,
preregistered evaluation, no hardcoded task answers). Status:

- Module 1 — arc_static/dsl.py: BUILT AND MERGED. Pure deterministic grid
  primitives, 12 tests. Verified: single primitives alone solve 19/400
  ARC-1 public TRAINING tasks.
- Module 2 — arc_static/search.py: BUILT AND MERGED. Deterministic depth-2
  program synthesis, verified-on-all-train-pairs or honest None, explicit
  budgets, 9 tests. Preregistered results at 0.5s/task budget: ARC-1
  public EVALUATION 2/400 pass@1 exact (prior baseline 1/400 — rung one
  "beat 1/400" is SUPPORTED); ARC-1 TRAINING 28/400.
- Module 1b — object-level primitives built on the existing frame_parser:
  NEXT. The training-vs-evaluation gap (28 vs 2) shows global grid
  transforms cap out; evaluation tasks demand object-level reasoning.
- Module 3 — loader + preregistered grader. Module 4 — arc_memory.
  Module 5 — your LLM hypothesis layer, which proposes DSL programs that
  the search engine must verify; your proposals never bypass verification.

Preregistered claims ladder (fixed harness, no tuning after results):
beat 1/400 ARC-1 eval [ACHIEVED at 2/400] → beat 10 percent → beat 0/120
ARC-2 eval. Every result recorded whatever it is.

## Your reply must contain

1. Acknowledgment of each Part 1 correction with proper classification.
2. The narrowest honest claim that module 2's completion demonstrates,
   and what it does NOT demonstrate.
3. Your design input for module 1b: which object-level primitives
   (built on ParsedFrame objects: color, cells, bounding box, centroid,
   shape signature) you predict will close the most evaluation-task gap,
   ranked, with your reasoning classified as hypothesis.
