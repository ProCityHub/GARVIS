# GARVIS Preregistration Ledger

Owner and final authority: Adrien D. Thomas (ProCityHub/GARVIS).

A preregistration is a public commitment to a protocol, analysis, and
decision rule **before** the data are examined. Entries are appended,
never edited or removed. Superseding a preregistration requires a new
entry and a RETRACTIONS.md record.

This ledger exists because a result that was not preregistered can be
silently reinterpreted after the fact. A preregistered negative result
is the most credible output the organization can produce.

## Vocabulary

| Terms | Reserved for |
|---|---|
| `SUPPORTED` / `NOT_SUPPORTED` | pre-registered empirical outcomes only |
| `PASS` / `FAIL` | software test and check outcomes |

These are never mixed. `.github/scripts/guard_check.py` enforces this in CI.

## Entry format

```
### P-XXX  <short title>
- **Hypothesis:** what is claimed
- **Protocol:** how it is tested
- **Decision rule:** what outcome counts as SUPPORTED vs NOT_SUPPORTED
- **Pre-registration date:** YYYY-MM-DD
- **Outcome:** SUPPORTED / NOT_SUPPORTED / PENDING
- **Outcome date:** YYYY-MM-DD or PENDING
- **Evidence:** path to frozen evidence artifact
- **Retraction:** R-XXX if retracted, or NONE
```

---

## Preregistrations

### P-001  Prime-OAB interference discriminator vs standard quantum mechanics
- **Hypothesis:** The Canonical Lattice Law interference discriminator
  (V_phi) predicts asymmetric-path double-slit outcomes more accurately
  than standard quantum mechanics (V_QM).
- **Protocol:** Run QASM circuits on IBM Quantum (ibm_fez, 156-qubit Heron
  r2) with asymmetric-path configurations. Compare residual error of V_phi
  vs V_QM on measured probability distributions (marginal p1, entropy).
- **Decision rule:** V_phi residual error < V_QM residual error at
  p < 0.05 across >= 3 independent job runs → SUPPORTED.
  Otherwise → NOT_SUPPORTED.
- **Pre-registration date:** 2026-07-XX
- **Outcome:** NOT_SUPPORTED
- **Outcome date:** 2026-07-XX
- **Evidence:** `research/quantum/prime_oab/evidence/ibm_fez_history.json`
  (frozen in FROZEN_FILES.txt)
- **Retraction:** NONE
- **Notes:** 7 IBM Quantum jobs completed on ibm_fez. The discriminator
  (V_phi) did not outperform V_QM on asymmetric-path data. The result is
  honest and is the organization's most credible finding. The hypothesis
  remains open pending improved formulations.

### P-002  Scalar phi formulation of the Canonical Lattice Law
- **Hypothesis:** The Canonical Lattice Law in its scalar form
  (C = O x A x B x phi) describes observed lattice coupling.
- **Protocol:** Self-audit against the exponentiated form
  (C = O^1 . A^(1/phi) . B^(1/phi^2)) and mathematical identity checks.
- **Decision rule:** Scalar form passes identity checks and is consistent
  with the exponentiated form → SUPPORTED. Otherwise → NOT_SUPPORTED.
- **Pre-registration date:** 2026-06-XX
- **Outcome:** NOT_SUPPORTED
- **Outcome date:** 2026-06-XX
- **Evidence:** Retraction R-003 (inherited organization-wide)
- **Retraction:** R-003
- **Notes:** The scalar form failed self-audit. Superseded by the
  exponentiated form with phi in the exponents. Enforced by
  `SCALAR_PHI_PATTERNS` in `.github/scripts/guard_check.py`.

### P-003  V16 instrument-validity gate
- **Hypothesis:** The V16 instrument can register a negative result.
- **Protocol:** Run the instrument with a known-false input and verify
  it reports FAIL.
- **Decision rule:** Instrument reports FAIL on known-false input →
  SUPPORTED. If it cannot report FAIL → NOT_SUPPORTED.
- **Pre-registration date:** 2026-06-XX
- **Outcome:** NOT_SUPPORTED
- **Outcome date:** 2026-06-XX
- **Evidence:** Retraction R-001 (originally in hypercubeheartbeat)
- **Retraction:** R-001
- **Notes:** The V16 instrument hard-coded `--observed 1.0 || true`,
  making it incapable of registering a negative result. An instrument
  that cannot fail is not measuring. This is the same defect addressed
  in the brain gate audit (2026-08-16) — all gates now bind --observed
  to job.status.

### P-004  hypercubeheartbeat public replication
- **Hypothesis:** The hypercubeheartbeat results can be independently
  replicated from public artifacts.
- **Protocol:** Publish all code, data, and job archives in a public
  repo. Attempt replication from scratch.
- **Decision rule:** Independent replication reproduces published
  results → SUPPORTED. If repo is deleted or artifacts are unreachable
  → NOT_SUPPORTED.
- **Pre-registration date:** 2026-06-XX
- **Outcome:** NOT_SUPPORTED
- **Outcome date:** 2026-08-16
- **Evidence:** N/A — repo deleted
- **Retraction:** R-002
- **Notes:** The hypercubeheartbeat repository was deleted, along with
  AGI and THUNDERBIRD. PREREGISTRATION.md, CITATION.cff, and retractions
  R-001/R-002 were hosted there and are no longer publicly reachable.
  This entry restores the preregistration record to a public repo.
  The evidence for P-001 survives in the GARVIS repo.
