# GARVIS Retractions Ledger

Owner and final authority: Adrien D. Thomas (ProCityHub/GARVIS).

A retraction is a public, permanent record that something previously stated
in this repository was wrong. Entries are appended, never edited or removed.
Superseding a retraction requires a new entry, not a rewrite.

This ledger exists because the alternative is silent correction, and a claim
that can be quietly withdrawn was never a claim.

## Vocabulary

| Terms | Reserved for |
|---|---|
| `PASS` / `FAIL` | software test and check outcomes |
| `SUPPORTED` / `NOT_SUPPORTED` | pre-registered empirical outcomes only |

These are never mixed. `.github/scripts/guard_check.py` enforces this in CI.

## Entry format

```
### R-XXX  <short title>
- **Status:** RETRACTED
- **Date:** YYYY-MM-DD
- **Retracted claim:** what was asserted
- **Why it was wrong:** the specific defect
- **Superseded by:** the correct statement, or NONE
- **Enforcement:** what now prevents recurrence
```

---

## Inherited organization-wide retractions

### R-003  Scalar Lattice Law formula
- **Status:** RETRACTED
- **Origin:** ProCityHub/hypercubeheartbeat, applies organization-wide
- **Retracted claim:** `C = O x A x B x phi` (phi as a scalar multiplier)
- **Why it was wrong:** the scalar form was not the tested formulation and
  did not survive self-audit
- **Superseded by:** `C = O^1 . A^(1/phi) . B^(1/phi^2)`, with phi in the
  exponents
- **Enforcement:** `SCALAR_PHI_PATTERNS` in
  `.github/scripts/guard_check.py` fails CI if the pattern is reintroduced

---

## GARVIS retractions

None recorded.

This section is empty because no GARVIS claim has yet been found false --
not because none can be. When one is, it goes here before anything else
proceeds.
