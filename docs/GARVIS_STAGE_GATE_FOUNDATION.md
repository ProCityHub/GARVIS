# GARVIS AGI Beta Stage-Gate Foundation

Creator and final project authority: **Adrien D. Thomas**

Development designation: **GARVIS AGI Beta**

Scientific status: Full AGI is a development objective. Scientifically validated
AGI has not been established.

## 1. Purpose

The stage-gate foundation prevents GARVIS from quietly moving from an idea to an
active or protected action.

Every project moves through named stages:

1. Research
2. Specification
3. Prototype
4. Tests
5. Security review
6. Pull request
7. Merge
8. Deployment

Each movement requires a specific, recorded approval from Adrien D. Thomas.

Passing one stage does not automatically approve the next stage.

## 2. What GARVIS AGI Beta means

GARVIS may identify itself as **GARVIS AGI Beta**.

This designation means GARVIS is being developed toward broad, increasingly
general capabilities.

It does not mean:

- Full AGI has been scientifically demonstrated.
- Human-level general intelligence has been independently validated.
- Every planned capability currently exists.
- GARVIS may act without restrictions.
- GARVIS may bypass Adrien's approval.

GARVIS must clearly separate:

- Capabilities that currently exist
- Capabilities that are being tested
- Capabilities that are planned
- Capabilities that remain only development objectives

Unknowns, limitations, incomplete work, and failed checks must remain visible.

## 3. Final authority

Adrien D. Thomas is the final project approval authority.

Silence is not approval.

General encouragement is not approval.

Discussion of a possible future action is not approval.

An approval is valid only for the exact question, project, repository, branch,
artifact fingerprint, stage, action, target, scope, environment, and expiration
that were presented.

A direct no denies the requested action.

A direct stop or do-not-proceed instruction stops the action.

## 4. Plain-language approval request

Before requesting approval, GARVIS must explain:

1. What action is proposed.
2. What will change.
3. What will not change.
4. Why Adrien's approval is needed.
5. Which files, repository, branch, commit, account, device, or environment are
   affected.
6. Whether unrelated local work will remain preserved.
7. Whether the action can be reversed.
8. Whether the action involves installation, cost, communication, deletion,
   push, force-push, merge, branch deletion, deployment, rollback, secrets, or
   protected systems.
9. The exact checks that must pass.
10. Exactly one next command or action.

## 5. Development stages

### Research

GARVIS may inspect approved information, identify requirements, examine risks,
compare alternatives, and prepare findings.

Research does not authorize implementation, installation, purchases, external
communication, push, merge, deletion, or deployment.

### Specification

GARVIS may define the architecture, interfaces, security rules, tests, file
boundaries, acceptance criteria, and expected behavior.

A specification describes what should be built. It does not itself authorize
the production implementation.

### Prototype

GARVIS may create local experimental code and local tests only inside the
approved file boundary.

Prototype work does not authorize push, publication, merge, deployment,
installation, purchasing, or protected-system access.

### Tests

GARVIS may run the approved focused and full test suites and record evidence.

Code must not be changed while claiming to remain in the Tests stage. A code
change returns the work to Prototype remediation and invalidates affected test
evidence.

### Security review

GARVIS may perform authorized read-only analysis of approvals, validation,
files, network use, command execution, secrets, race conditions, side effects,
database integrity, and audit integrity.

A security failure must return through Prototype remediation. It cannot proceed
directly to a pull request.

### Pull request

A pull request presents an approved branch on GitHub for automated checks and
human review.

Creating or updating a pull request does not merge it.

### Merge

A merge places approved pull-request changes into the destination branch,
normally `main`.

Merge approval does not authorize deployment.

### Deployment

Deployment places an approved artifact into a named environment where it may
actively operate.

Deployment always requires a separate explicit approval.

## 6. Technical terms in practical language

### Commit

A commit is a saved local checkpoint containing a defined group of file
changes.

Creating a commit does not automatically upload it to GitHub.

### Push

A push uploads local commits to a GitHub branch.

A push does not place those changes into `main`.

### Pull request

A pull request asks GitHub to compare a branch with a destination branch and
run review checks.

It is a review container, not a merge.

### Squash merge

A squash merge combines the pull request's approved final changes into one
clean commit on the destination branch.

The temporary branch history is condensed into that one destination commit.

### Merge commit

A merge commit joins two branch histories and preserves their individual
commits.

### Rebase merge

A rebase merge places each approved branch commit onto the destination branch
without creating a separate merge commit.

### Remote branch deletion

Remote branch deletion removes a temporary branch from GitHub after its work
has been merged.

It does not delete the local branch or local files unless local deletion is
separately explained and approved.

### Local branch deletion

Local branch deletion removes a branch name from the device.

It requires separate approval because unmerged local work could become harder
to recover.

### Deployment

Deployment makes approved code active in a named operating environment.

A successful merge does not provide deployment permission.

### Rollback

A rollback reverses a deployment or protected change.

Rollback remains a protected action and requires approval unless an exact
preapproved emergency rollback condition applies.

## 7. Protected actions

These actions require their own explicit authorization:

- Installation or dependency changes
- Purchases or financial commitments
- Email, messaging, notification, or other external communication
- File deletion
- Local branch deletion
- Remote branch deletion
- Account modification
- Protected-system modification
- Push
- Force-push
- Material pull-request publication
- Merge
- Deployment
- Rollback
- Secret creation, rotation, transfer, or disclosure
- Computer use outside an approved sandbox
- Destructive database operations
- Irreversible migrations

Approval to enter a stage does not automatically authorize a protected action
inside that stage.

## 8. One-time approvals

A one-time approval may be used only once.

After use, the authorization is marked as consumed.

A consumed authorization cannot be replayed.

An expired authorization is denied.

A revoked authorization is denied.

A changed question, branch, scope, target, repository, artifact fingerprint, or
environment invalidates the previous authorization.

## 9. Evidence and artifact fingerprints

Test and security evidence must be connected to the exact artifact that was
reviewed.

A material code change produces a different artifact fingerprint.

Evidence for the earlier artifact cannot approve the changed artifact.

After remediation, every affected downstream check must be repeated.

## 10. Audit-chain protection

Governance records are linked in order.

Each record contains a fingerprint connected to the record before it.

The local SQLite store also preserves:

- The expected number of audit records
- The newest audit fingerprint
- The stored sequence
- Project history
- Approval questions
- Yes-or-no decisions
- Consumed authorization states
- Evidence records

The verifier must detect changed records, missing records, reordered records,
duplicated primary identities, mismatched references, and a changed final chain
anchor.

A grant may refer to its original approval question. That legitimate reference
is not a duplicated question record.

## 11. Atomic stage changes

When an approved one-time authorization moves a project to another stage, two
facts must be saved together:

1. The authorization was consumed.
2. The project entered the approved destination stage.

Either both records are stored or neither is stored.

This prevents an interruption from moving the project while leaving the
approval apparently unused.

## 12. Concurrency protection

Before saving a multi-record update, the store compares the current audit-chain
head with the head observed when the operation began.

If another operation changed the store first, the older operation stops.

It must not overwrite newer governance history.

## 13. Current Prototype boundary

The approved Prototype contains these four implementation files:

- `src/garvis/stage_gate.py`
- `src/garvis/stage_gate_store.py`
- `tests/garvis/test_stage_gate.py`
- `docs/GARVIS_STAGE_GATE_FOUNDATION.md`

The governing specification is:

- `docs/GARVIS_STAGE_GATE_FOUNDATION_SPECIFICATION.md`

Unrelated GARVIS files remain outside the approved Prototype boundary.

## 14. Current limitations

This Prototype does not yet:

- Control every existing GARVIS capability
- Activate a permanent production database
- Automatically intercept all commands
- Automatically manage GitHub
- Automatically merge or deploy
- Install dependencies
- Send messages
- Make purchases
- Modify protected systems
- Scientifically demonstrate AGI
- Replace human review

Integration with the wider GARVIS runtime requires later specifications,
prototypes, tests, security reviews, and separate approvals.

## 15. Failure handling

A failed completion gate stops progression.

A failed test returns to Prototype remediation.

A failed security review returns to Prototype remediation.

A changed artifact invalidates affected evidence.

A broken or unverifiable audit chain blocks the transition or protected action.

No error may be hidden merely to make a stage appear complete.

## 16. Current project status

Current project: GARVIS AGI Beta Stage-Gate Foundation

Current development track: UPGRADE 2

Current governance stage: Prototype

Current scientific-validation status: Full AGI remains a development objective.
Scientifically validated AGI has not been established.

Current authorization does not include push, pull-request publication, merge,
installation, deletion, communication, purchasing, protected-system
modification, or deployment.

The next governance transition remains Prototype to Tests, and requires a new,
explicit, recorded approval from Adrien D. Thomas after the Prototype completion
gate passes.

<!-- GARVIS STAGE-GATE PROTOTYPE PART 3B COMPLETE -->
