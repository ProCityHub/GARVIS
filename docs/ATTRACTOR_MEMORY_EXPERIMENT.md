# GARVIS Attractor-Memory Prototype

Author: Adrien D. Thomas

Status: deterministic software experiment

Scientific boundary: this prototype does not establish consciousness,
sentience, identity persistence, true AGI, biological equivalence,
quantum validation, or a physical law.

## Purpose

Build a Hopfield-style associative-memory component that:

1. Stores bipolar patterns in distributed interaction weights.
2. Accepts complete, partial, or noisy cues.
3. Iteratively settles toward a stable state.
4. Reports convergence, recovery accuracy, energy, and failures.

The network weights are the storage mechanism. This must not be described
as memory existing without storage.

## State representation

Stored patterns contain only -1 and +1.

Recall cues may also contain 0, representing an erased or unknown value.

All patterns must have the same length.

## Weight construction

For stored patterns x_k of dimension n:

W = (1 / n) * sum of outer_product(x_k, x_k)

The weight matrix must be symmetric.

Every diagonal value must be zero.

Recall must use the weight matrix rather than a filename, database key,
pattern identifier, or lookup index.

## Settling rule

Use deterministic asynchronous neuron updates.

For each neuron:

field = sum(W[i][j] * state[j])

If field is positive, set the neuron to +1.
If field is negative, set the neuron to -1.
If field is zero, preserve its current value.
If both the field and current value are zero, use +1.

One sweep updates every neuron once.

The network converges when one complete sweep produces no changes.

A finite maximum number of sweeps is mandatory.

## Energy

For a fully bipolar state:

E = -0.5 * state_transpose * W * state

Energy should not increase during deterministic asynchronous settling.

This is a software invariant, not evidence of physical energy.

## Required measurements

Each recall result must report:

- final state;
- converged or not converged;
- number of sweeps;
- total state changes;
- energy trace;
- exact target match when a target is supplied;
- Hamming distance when a target is supplied;
- warnings or failure reason.

## Required tests

The implementation must test:

1. Empty-pattern rejection.
2. Inconsistent-dimension rejection.
3. Invalid-value rejection.
4. Symmetric weights and zero diagonal.
5. Exact recall from complete stored cues.
6. Recall from partial cues.
7. Recall from controlled bit-flip noise.
8. Deterministic results using the same seed.
9. Bounded non-convergence handling.
10. Non-increasing asynchronous energy.
11. False-attractor reporting.
12. Capacity and interference with multiple patterns.

Successful, failed, and ambiguous recalls must all remain visible.

## Baselines

The experiment must compare attractor recall against:

1. An explicit nearest-pattern lookup baseline.
2. A no-memory baseline that fills erased bits deterministically without
   learned associative settling.

The comparison is descriptive unless a separate experiment is preregistered.

## Engineering boundary

The first implementation will use only the Python standard library.

Proposed module:

src/garvis/attractor_memory.py

Proposed tests:

tests/garvis/test_attractor_memory.py

The memory module must not perform:

- filesystem writes;
- network access;
- subprocess execution;
- Git operations;
- secret access;
- model calls;
- external actions.

## Future GARVIS integration

A later integration may supply attractor output as optional context.

Recalled content must include provenance and confidence and must never be
treated as unquestioned truth.

Attractor memory remains separate from session history, repository evidence,
scientific claims, approvals, and external-action authority.
