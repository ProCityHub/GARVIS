# GARVIS Epistemic Filing System v1

Authority: Adrien D. Thomas / ProCityHub.

## Purpose

GARVIS must not treat every failed threshold as the same kind of error.

A software failure belongs in the operational error registry. A statement that
is not yet established as fact belongs in the epistemic claim registry. Both are
preserved, classified, reviewable, and auditable.

"Not verified" does not automatically mean "false", and it does not mean the
record should be deleted.

## Two filing houses

Operational failures are filed as syntax, test, typecheck, lint, runtime, data,
security, governance, or unknown. Their lifecycle is:

`open -> triaged -> fix_in_progress -> resolved`

Epistemic claims are filed as verified, supported, provisional, hypothesis,
speculative, symbolic, anomaly, unknown, contradicted, retracted, or
identity_draft.

Every claim preserves its exact statement, domain, scope, confidence, evidence,
counterevidence, failure conditions, permitted wording, prohibited wording,
review state, and revision history.

## Governance rules

1. A verified claim requires supporting evidence.
2. Counterevidence blocks unqualified fact wording without deleting the claim.
3. A scientific fact additionally requires reproducible supporting evidence.
4. An identity draft may be expressed only with its provisional scope.
5. Status changes require an actor and reason and are appended to history.
6. Claims and software errors remain separate, but an error may link to claims.
7. No classification bypasses tests, type checks, review, or external-action authority.

## Heartbeat mapping

- 0.0: preserve the exact statement, source, or failure.
- 0.2: segment the statement and identify its domain.
- 0.6: classify evidence, counterevidence, confidence, and coherence.
- 1.0: generate wording permitted by the current classification.
- 1.4: compare the intended claim or result with observed evidence.
- 1.6: archive the record, revision history, and next review condition.

This module is a filing and governance foundation. It does not automatically
declare claims true, scientific, conscious, AGI, or resolved.
