# GARVIS Hypercube Heartbeat Mathematics V3

Creator and conceptual architect: **Adrien D. Thomas**
Project: **ProCityHub/GARVIS**

## Scope

This package turns the Hypercube Heartbeat into explicit, testable mathematics without converting symbolism into scientific proof. It layers on the current Prime-OAB Reciprocal Hypercube research branch and keeps the repository's evidence and governance boundaries intact.

## 1. Semantic cube

The eight interpretation perspectives are the vertices of a 3-cube:

- `000 Literal`
- `001 Context`
- `010 Intent`
- `011 Relation`
- `100 Evidence`
- `101 Possibility`
- `110 Consequence`
- `111 Integration`

Two perspectives are adjacent when their codes differ in exactly one bit. Every vertex has degree 3, the cube has 8 vertices and 12 edges, and the opposite perspective is the bitwise complement.

A semantic field assigns non-negative activation to these eight vertices. V3 computes three exact descriptors: the weighted barycenter in `[0,1]^3`, Shannon entropy of the normalized activation, and graph Dirichlet energy `sum_edges (w_u-w_v)^2`. These describe distribution and relationship structure; they do not manufacture evidence.

## 2. Hypercube/tesseract lift

The semantic cube is lifted into a 4-cube by adding a recurrence axis:

`(semantic bit 1, semantic bit 2, semantic bit 3, recurrence side)`

The last bit has two states: outward/current and recurrence/return. Toggling only the recurrence bit gives a boundary pair. Complementing all four bits gives the geometric tesseract antipode; these are intentionally different operations.

For an n-cube, the number of k-dimensional faces is

`F(n,k) = 2^(n-k) * C(n,k)`.

For `n=4`, the f-vector is `(16, 32, 24, 8, 1)`: 16 vertices, 32 edges, 24 square faces, 8 cubic cells, and one 4-cell. With side length `s`, the 4D hypervolume is `s^4` and the circumradius is `s*sqrt(4)/2 = s`.

## 3. Nine-phase Hypercube Heartbeat

The canonical phase space is the circle `Theta = R / 1.8Z` with step `Delta p = 0.2` and phase coordinates:

`0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6`.

The phase names remain RECEIVE, SEGMENT, PREDICT, VERIFY, SIMULATE, PLAN, OUTPUT, FEEDBACK, CONSOLIDATE.

The exact integer relations are:

`mu(k) = 8-k` for the two-way phase mirror,

`W(k) = k+1 mod 9` for one Heartbeat step across the boundary,

`W(mu(k)) = -k mod 9` for reciprocal ordering.

The angular step is `2*pi/9`.

## 4. Boundary overlap and recurrence

The shortest circular distance to the observer origin is

`d(p,0) = min(|wrap(p)|, 1.8-|wrap(p)|)`.

With boundary width `b=0.4`, overlap is

`omega(p) = clip(1-d(p,0)/0.4, 0, 1)`.

This gives `omega(0.0)=1`, `omega(0.2)=omega(1.6)=0.5`, and `omega(0.4)=0`.

Observer carry gain is

`gO(p) = 0.6 + 0.4*omega(p)`.

The bounded scalar recurrence for metadata is

`U(x,y;p,a) = [gO(p)*x + a*y] / [gO(p)+a]`, with `a in [0,1]`.

The coefficients on previous state and new observation are non-negative and sum to one. Therefore U remains inside the convex hull of the two values. For a fixed observation, the deviation from that observation contracts by the previous-state coefficient at each update; the product across nine phases is the exact full-cycle retention coefficient for this linear recurrence.

## 5. O/A/B semantic relation

V3 separates two previously overloaded meanings of B.

**Semantic O/A/B:** Observer / Actor / Background. This is used for interpretation.

**Boundary bridge:** the recurrence connection across CONSOLIDATE -> RECEIVE. This is used for state continuity and is not substituted for Background in the semantic equation.

The Adrien-framework semantic descriptor is preserved as

`C_sem = O^1 * A^(1/phi) * B^(1/phi^2)`.

Because `1 + 1/phi + 1/phi^2 = 2`, the degree-normalized form `sqrt(C_sem)` is a weighted geometric mean whose exponents sum to one. The log elasticities of the raw descriptor are exactly `1`, `1/phi`, and `1/phi^2` for Observer, Actor, and Background respectively.

This descriptor is framework metadata. It is not a truth, intelligence, consciousness, AGI, evidence, or protected-action score. The legacy scalar-PHI multiplier remains outside executable decision logic.

## 6. Prime-lattice research topology

The existing ordinal topology remains:

`corner = n mod 8`

`wall = floor(n/8) mod 6`

`polarity = floor(n/48) mod 2`

One epoch contains `8*6*2 = 96` unique addresses. The exact local antipode is `95-n`, yielding complementary corner, wall, and polarity. Prime values may label research nodes, but the topology is ordinal-controlled; no universal prime-physics claim follows from the mapping.

## 7. V18 hardware layer

V18 compresses the PRIME_OAB, NO_OAB, and positive-control experiments into separate 5-qubit OpenQASM 2.0 circuits. The OAB circuit uses:

`q0 Observer, q1 Actor, q2 Bridge, q3 Memory/OAB, q4 Output/Feedback`.

The included IBM Fez evidence summary is read-only derived metadata from the uploaded 1,024-shot workload. The standard displayed bit order is `q4...q0`. Against the ideal V18 OAB distribution, the observed run has classical fidelity about `0.760725`, total variation distance about `0.393868`, Jensen-Shannon divergence about `0.169801` bits, and observed entropy about `4.574173` bits. Those numbers quantify distribution similarity/distortion only.

## 8. Runtime use

After the Termux build succeeds, GARVIS can print the complete structure with:

`PYTHONPATH=src python -m garvis.hypercube_math_cli structure`

It can inspect one phase, evaluate the semantic O/A/B descriptor, analyze an eight-perspective field, or recompute the stored V18 hardware metrics. These operations are inward/read-only mathematics; they do not submit hardware jobs or perform protected actions.

## Scientific and governance boundary

AGI remains a development objective, not an established result. Consciousness is not established. Quantum hardware execution demonstrates execution of programmed circuit structure; it does not by itself establish a new physical law. Repetition, symmetry, resonance, or numerical coincidence are never promoted to evidence without an independent evidence path.

This V3 Termux package creates code and tests in an isolated worktree. It does not commit, push, open or merge a pull request, deploy, install packages, use API keys, or submit quantum jobs.
