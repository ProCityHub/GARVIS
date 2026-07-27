# GARVIS Universal AI Router V1.1

Creator / conceptual architect: **Adrien D. Thomas**

## Purpose

GARVIS remains the orchestrator. External AI systems are replaceable candidate
reasoning organs, not identities, truth authorities, or execution authorities.

V1.1 provides:

- normalized provider identity metadata,
- explicit provider capability metadata,
- deterministic health-aware candidate ordering,
- Hypercube perspective scheduling,
- fail-closed handling of unknown provider identities,
- explicit distinction between installed AI apps and programmable adapters.

## Hypercube ownership boundary

The initial scheduling invariant preserves:

- `001 Context` -> GARVIS,
- `100 Evidence` -> GARVIS evidence,
- `111 Integration` -> GARVIS.

External provider output remains `CANDIDATE_ONLY`.

## Provider identity security

Known provider families are resolved explicitly. Unknown provider/model
identifiers fail closed with no supported adapter and are not programmable.

Environment-variable presence is availability metadata only. It does not
establish identity, capability, evidence, trust, or authority.

## Health boundary

Provider health is operational routing evidence only. Recorded success,
failure count, or block status must never be interpreted as proof that a
provider answer is true.

## Security review

The standalone prototype completed:

- prototype validation,
- adversarial tests,
- fail-closed identity hardening,
- security review rerun with 0 CRITICAL and 0 HIGH findings.

The remaining environment-trust note is retained as an architectural boundary:
environment configuration may signal availability only.

## Scientific / project boundary

This work supports the GARVIS AGI research program. It does not establish AGI,
consciousness, singularity, or a new physical law.

## Governance

Adrien-Approval: **APPROVE GARVIS UNIVERSAL AI ROUTER V1.1 PR STAGE**

This approval authorizes integration testing, commit, push, and draft pull
request creation for this artifact only.

Merge, deployment, real provider calls, purchases, messaging, deletion,
installation, and device-control permissions are not authorized.
