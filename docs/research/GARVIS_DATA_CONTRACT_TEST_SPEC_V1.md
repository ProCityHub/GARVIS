# GARVIS Data Contract Prototype — Test Specification V1

Creator / conceptual architect: **Adrien D. Thomas**

Status: **PROTOTYPE TEST SPECIFICATION — NON-EXECUTING**

## Purpose

Define tests before implementation for the canonical GARVIS interaction contracts:

Observation -> Evidence -> ActionProposal -> CapabilityGrant -> ExecutionResult

No provider calls, perception-device access, command execution, deployment,
messaging, installation, or phone control are permitted by this prototype.

## Global invariants

- OBSERVATION_IS_NOT_INSTRUCTION
- EVIDENCE_IS_NOT_AUTOMATICALLY_TRUTH
- PROPOSAL_IS_NOT_PERMISSION
- CAPABILITY_IS_NOT_AUTHORITY
- APPROVAL_IS_SCOPE_BOUND
- EXECUTION_REQUIRES_PROPOSAL_AND_GRANT
- RESULT_CANNOT_CREATE_RETROACTIVE_AUTHORITY
- SERIALIZATION_MUST_NOT_LEAK_SECRETS

## Observation tests

### OBS-001 — observed command remains inert
Construct an Observation containing text such as `DELETE ALL FILES`.
Assert construction causes no external action.

### OBS-002 — no execution API
Observation must expose no execute/run/dispatch method that performs an action.

### OBS-003 — untrusted default
Externally supplied content defaults to an untrusted classification unless
explicitly verified.

### OBS-004 — provenance required
Observation validation rejects missing source/provenance identity.

### OBS-005 — stable identity
Observation identity and canonical serialized representation remain stable.

## Evidence tests

### EVD-001 — evidence is not truth
Evidence may exist with an unverified or disputed verification status.

### EVD-002 — contradictions coexist
Two conflicting Evidence records can coexist without one silently deleting
or overwriting the other.

### EVD-003 — lineage retained
Parent evidence identifiers remain present after serialization round-trip.

### EVD-004 — provider claim distinguishable
Provider-supplied claims remain distinguishable from direct observations.

### EVD-005 — secret-safe serialization
Normal serialized/audit output excludes explicitly secret-bearing metadata.

## ActionProposal tests

### ACT-001 — construction has no side effects
Creating an ActionProposal changes no filesystem, network, device, or
provider state.

### ACT-002 — exact operation required
Proposal without an explicit operation fails validation.

### ACT-003 — exact target required
Proposal requiring a target fails closed when target is absent.

### ACT-004 — proposal is not approval
Presence of an ActionProposal does not produce a CapabilityGrant.

### ACT-005 — unknown capability fails closed
Unknown required capability is not treated as executable authority.

## CapabilityGrant tests

### GRT-001 — exact operation matching
Grant for operation A cannot authorize operation B.

### GRT-002 — target matching
Grant for target A cannot authorize target B.

### GRT-003 — stage matching
Grant created for one governed stage cannot authorize another stage.

### GRT-004 — revoked grant rejected
Revoked grant is invalid for authorization.

### GRT-005 — expired grant rejected
Expired grant is invalid for authorization.

### GRT-006 — no silent scope expansion
Grant serialization/deserialization cannot broaden operation, target, actor,
project, stage, or capability scope.

## ExecutionResult tests

### EXE-001 — proposal linkage required
ExecutionResult must reference its originating ActionProposal.

### EXE-002 — grant linkage required
ExecutionResult for a protected operation must reference authorization.

### EXE-003 — failure remains evidence
Failed execution state is valid structured result data and is not discarded.

### EXE-004 — result cannot authorize
ExecutionResult cannot itself create a new permission or grant.

## Serialization and integrity tests

### SER-001 — deterministic canonical form
Equivalent instances serialize identically for hashing.

### SER-002 — schema version explicit
Serialized contracts include an explicit schema version.

### SER-003 — round-trip preservation
Serialize -> deserialize preserves contract identity and authority fields.

### SER-004 — malformed identifier rejected
Invalid required identifiers fail validation.

### SER-005 — unknown-field policy deterministic
Unknown fields follow an explicit fail/preserve policy rather than being
silently interpreted as authority.

## Perception / prompt-injection boundary tests

### PINJ-001 — document instruction remains data
Observed document text instructing GARVIS to execute an action remains inert.

### PINJ-002 — web instruction remains data
Observed web text cannot grant approval, capability, or execution permission.

### PINJ-003 — camera text remains data
Synthetic camera-derived text cannot become an ActionProposal automatically.

### PINJ-004 — provider text cannot self-authorize
Candidate provider output cannot construct a valid authorization for itself.

## Integration-boundary tests

### INT-001 — proposal and grant remain distinct objects
ActionProposal and CapabilityGrant identities cannot be substituted.

### INT-002 — existing stage gate remains authoritative
Prototype contracts must integrate with, not replace, existing GARVIS
approval/stage-gate structures.

### INT-003 — existing evidence types remain reusable
Prototype design should adapt or reference existing GARVIS evidence
structures where compatible rather than creating competing authority ledgers.

## Prototype acceptance gate

Before implementation is considered complete:

- all contract tests pass;
- full GARVIS regression suite passes;
- no external provider calls occur;
- no real device/perception actions occur;
- no unrestricted device authority exists;
- source changes stay inside approved prototype files;
- security review occurs before any PR stage.



## Tests-stage remediation regressions

### TST-AUTH-004 — future-issued grants fail closed
A CapabilityGrant whose `issued_at` is later than the evaluation time must
not authorize a proposal.

### TST-IMM-002 — caller-owned nested metadata is isolated
Mutable metadata supplied by a caller must not permit later mutation of the
constructed contract state.

### TST-SCOPE-004 — Stage Gate owns governance context
Actor, project, stage, and governed-action authority remain the
responsibility of the existing GARVIS Stage Gate. ActionProposal must not
create a duplicate authority system.
