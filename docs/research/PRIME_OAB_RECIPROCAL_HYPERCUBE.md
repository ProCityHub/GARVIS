# Prime-OAB Reciprocal Hypercube

Creator / conceptual architect: **Adrien D. Thomas (ProCityHub/GARVIS)**.

## Purpose

This directory converts the quantum research program into repository code rather
than a chat-only concept.

The architecture is:

`experience -> Heartbeat -> OAB retention -> reciprocal return -> changed next state`

The intended research question is whether this recurrence improves measurable
memory, correction, planning, transfer, and long-horizon behaviour.

It is not an AGI, consciousness, singularity, or theory-of-everything claim.

## Canonical Heartbeat

| Phase | Coordinate | Mirror |
|---|---:|---|
| RECEIVE | 0.0 | CONSOLIDATE |
| SEGMENT | 0.2 | FEEDBACK |
| PREDICT | 0.4 | OUTPUT |
| VERIFY | 0.6 | PLAN |
| SIMULATE | 0.8 | SIMULATE |
| PLAN | 1.0 | VERIFY |
| OUTPUT | 1.2 | PREDICT |
| FEEDBACK | 1.4 | SEGMENT |
| CONSOLIDATE | 1.6 | RECEIVE |

Discrete mirror index:

`mu(i) = 8 - i`

OAB step:

`W(i) = (i + 1) mod 9`

Combined reciprocal ordering:

`W(mu(i)) = -i mod 9`

This is an architectural reciprocal ordering, not physical reversal of time.

## Prime Lattice

For zero-based prime ordinal `n`:

`corner = n mod 8`

`wall = floor(n/8) mod 6`

`polarity = floor(n/48) mod 2`

`epoch = floor(n/96)`

One epoch contains:

`8 * 6 * 2 = 96`

unique addresses.

Prime value identifies the node. Prime ordinal controls topology.

For the first epoch the prime identities run from `2` through `503`.

Their 95 consecutive gaps telescope to:

`503 - 2 = 501`.

## Two-way lattice mirror

Within an epoch:

`J(n) = 95 - n`

and the topology complements:

`corner -> 7-corner`

`wall -> 5-wall`

`polarity -> 1-polarity`

The finite first-epoch Prime Mirror Defect is:

`D_n = 505 - (p_n + p_(95-n))`

for `n=0..47`.

For the current 96-prime window:

- min defect = 0
- max defect = 73
- sum defects = 2201

This is an exact property of this finite sequence and must not be presented as a
universal number-theory law.

## OAB

OAB captures the full nine-role outward vector.

Return roles are seeded in complementary order:

`RECEIVE <- CONSOLIDATE`

`SEGMENT <- FEEDBACK`

`PREDICT <- OUTPUT`

`VERIFY <- PLAN`

`SIMULATE <- SIMULATE`

and vice versa.

The residual between the seeded return and the observed return remains a vector.
It is deliberately **not reduced to a scalar intelligence/truth/consciousness score**.

## Quantum history

The `qasm/` directory preserves the V4-V12 progression used to develop the current
model. The current hardware-oriented design is the V12 128Q mirror field; the 5Q
runnable twin is the compressed experiment that produced a clean real-QPU signal.

The downloaded IBM workload history is stored as derived, read-only metadata in
`evidence/ibm_fez_history.json`.

The evidence supports claims about execution of the programmed circuits only.

## Retraction boundary

The organization-wide scalar expression `C = O x A x B x phi` is retracted in
`RETRACTIONS.md` and is not used here.

No PHI or Lattice-family scalar is an AGI, truth, consciousness, or decision score.

## Governance

This code is inward-only.

It has:

- no IBM credentials;
- no QPU submission code;
- no network calls;
- no filesystem writes outside explicit caller-directed file reading;
- no protected external action.

Future live IBM execution belongs behind GARVIS StageGate authorization, where
credentials authenticate and an exact scoped grant authorizes the action.
