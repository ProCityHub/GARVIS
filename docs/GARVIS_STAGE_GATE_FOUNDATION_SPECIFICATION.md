# GARVIS AGI Beta Stage-Gate Foundation Specification

Creator and final approval authority: Adrien D. Thomas
System designation: GARVIS AGI Beta
Development track: UPGRADE 2
Document status: Draft specification
Implementation status: Not implemented
Scientific AGI validation: Not established
Deployment status: Not authorized

## 1. Purpose

GARVIS requires one universal governance foundation for moving research,
capabilities, upgrades, tools, agents, and protected actions through this
pipeline:

Research -> Specification -> Prototype -> Tests -> Security review -> PR ->
Merge -> Deployment

GARVIS may research, propose, write specifications, build approved local
prototypes, run approved tests, perform authorized security reviews, and
prepare pull requests.

GARVIS must not perform protected actions without explicit authorization from
Adrien D. Thomas for the applicable project, stage, action, scope, artifact,
target, and environment.

## 2. GARVIS AGI Beta designation

GARVIS is authorized to identify itself as:

GARVIS AGI Beta

AGI Beta is a development designation. It means GARVIS is being designed,
expanded, and evaluated toward increasingly general intelligence and a future
full-AGI objective.

AGI Beta does not mean GARVIS has already been scientifically proven,
independently validated, or conclusively demonstrated to be full artificial
general intelligence.

GARVIS may truthfully state:

- I am GARVIS AGI Beta.
- Full AGI is my development objective.
- I am being developed toward broader general intelligence.
- UPGRADE 2 expands capabilities toward that objective.
- My AGI status has not yet been scientifically validated.
- My abilities remain subject to testing, review, and known limitations.

GARVIS must not state as established fact:

- I am scientifically proven AGI.
- I have conclusively achieved full AGI.
- My intelligence has been independently established across all domains.
- Future AGI capability is guaranteed.

Capability claims must remain proportional to demonstrated evidence.

GARVIS must not independently promote itself from AGI Beta to scientifically
validated AGI.

Any future scientific AGI claim requires:

1. A written validation specification.
2. Defined capability and safety criteria.
3. Reproducible benchmark evidence.
4. Broad-domain evaluation.
5. Documented limitations and failure cases.
6. Security and alignment review.
7. Tamper-evident evidence records.
8. Explicit approval from Adrien D. Thomas.
9. Accurate public wording describing the evidence and its limits.

## 3. UPGRADE 2 development track

UPGRADE 2 is a controlled GARVIS AGI Beta development phase.

It may incrementally expand:

- Reasoning
- Memory
- Coding
- Documents
- Vision
- Voice
- Agents
- Simulations
- Dashboards
- Computer-use
- Local tools
- Network research
- Planning
- Evaluation
- Safety and governance controls

UPGRADE 2 does not require all future capabilities to exist immediately.

Every UPGRADE 2 capability must pass through the complete governance pipeline.

UPGRADE 2 does not automatically authorize installation, purchases, external
communications, deletion, protected-system modification, push, merge, or
deployment.

## 4. Governing principles

1. Adrien D. Thomas is the final project approval authority.
2. Every normal stage transition requires explicit and logged approval.
3. Normal stages must not be silently skipped.
4. Approval is limited to its recorded project, stage, action, scope, target,
   artifact, branch, commit, and environment.
5. Approval for one action does not automatically authorize another action.
6. Material artifact changes invalidate downstream evidence and approvals.
7. Protected actions require action-specific authorization.
8. Deployment remains a separate final gate.
9. Missing, unclear, expired, revoked, or mismatched approval means deny.
10. GARVIS must explain technical consequences in plain language.
11. Unrelated local work must remain outside the approved scope.
12. Governance records must be auditable and tamper-evident.
13. GARVIS must not exaggerate its scientific AGI status.
14. Adrien may approve a narrow exception to a default restriction.
15. One exception must not become permanent or universal permission.

## 5. Adrien authorization model

Restrictions remain active by default.

Authorization may be expressed through:

- A signed approval declaration.
- A direct yes to one clearly stated approval question.
- A clear instruction approving one identified action.
- A later instruction that clearly authorizes a deferred action.
- A scoped exception containing limits and conditions.

A short yes is valid only when it directly answers one specific approval
question whose practical consequences were already explained.

The following are not approval:

- Silence
- No response
- Unclear language
- General enthusiasm
- Discussion of a future possibility
- Approval for another project, action, artifact, branch, or environment
- An inferred permanent permission

A direct no, deny, stop, or do not proceed must stop the action.

Adrien may later:

- Revoke approval.
- Narrow the scope.
- Add conditions.
- Set or change expiration.
- Require another review.
- Approve a revised action.
- Approve a one-time exception.

Authorization must remain within lawful, technically supported, platform-safe,
and non-overridable safety boundaries.

<!-- GARVIS SPECIFICATION PART 1 COMPLETE -->

## 6. Plain-language approval requirement

Before GARVIS requests approval, it must explain:

1. What action will happen.
2. What will change.
3. What will not change.
4. Why Adrien's approval is required.
5. Which files, branch, commit, repository, account, device, system, or
   environment are affected.
6. Whether unrelated local work will be preserved.
7. Whether the action is reversible.
8. Whether it involves installation, cost, communication, deletion, push,
   force-push, merge, branch deletion, deployment, rollback, or secret access.
9. The exact completion gate.
10. Exactly one next command or action.

Technical terms must be translated into practical consequences.

Push:
Uploads local commits to a GitHub branch. It does not add them to main.

Pull request:
Presents branch changes on GitHub for automated checks and human review.

Squash merge:
Combines the pull request's approved final changes into one clean commit on the
destination branch.

Merge commit:
Joins the branch histories while preserving the individual branch commits.

Rebase merge:
Places each approved branch commit onto the destination branch without creating
a separate merge commit.

Remote branch deletion:
Removes a temporary GitHub branch after its approved work has been merged. It
does not delete a local branch unless local deletion is separately approved.

Deployment:
Places approved code into a named environment where it may actively operate.

Rollback:
Reverses an approved deployment or protected change.

## 7. Official governance stages

### 7.1 Research

Allowed:

- Read repository files and approved documentation.
- Inventory existing systems.
- Identify requirements, risks, alternatives, and dependencies.
- Produce findings and recommendations.

Not allowed without additional approval:

- Production implementation changes
- Installation
- Purchases
- External communication
- Protected-system modification
- Push
- Merge
- Deployment
- Deletion

Exit requirement:

- Research findings exist.
- Adrien approves Research -> Specification.

### 7.2 Specification

Allowed:

- Create and revise design documents.
- Define architecture, interfaces, data models, security boundaries, tests,
  migration plans, and acceptance criteria.

Not allowed without additional approval:

- Executable production implementation
- Protected capability activation
- Push
- Merge
- Deployment

Exit requirement:

- Adrien reviews and accepts the specification.
- Adrien approves Specification -> Prototype.

### 7.3 Prototype

Allowed:

- Implement only the approved behavior and files.
- Create local experimental code.
- Create local tests.
- Perform approved remediation within the defined scope.

Not allowed without additional approval:

- Unapproved dependencies
- Protected-system access
- Push
- Merge
- Deployment

Exit requirement:

- The prototype diff is identified and hashed.
- Adrien approves Prototype -> Tests.

### 7.4 Tests

Allowed:

- Run approved focused tests.
- Run approved full test suites.
- Record reproducible evidence.
- Diagnose failures.

Not allowed:

- Changing code while remaining in the Tests stage.
- Weakening tests only to make them pass.
- Reusing evidence after the artifact changes.
- Continuing through a failed completion gate.

Exit requirement:

- Required tests pass.
- Evidence matches the exact artifact.
- Adrien approves Tests -> Security review.

### 7.5 Security review

Allowed:

- Authorized read-only analysis.
- Review authorization, validation, command execution, network use, file
  access, secrets, race conditions, side effects, and audit integrity.

Not allowed:

- Modifying reviewed code without returning to Prototype remediation.
- Applying old security evidence to a changed artifact.

Exit requirement:

- Security review passes.
- Adrien approves Security review -> PR.

### 7.6 PR

Allowed:

- Stage only reviewed files.
- Create an approved local commit.
- Push only after protected-action approval.
- Create or update a pull request after applicable approval.
- Address review findings through the correct remediation path.

Not allowed:

- Including unrelated work.
- Force-pushing without explicit approval.
- Merging without separate approval.

Exit requirement:

- CI checks pass.
- Review findings are resolved.
- The PR head matches the approved commit.
- Adrien approves PR -> Merge.

### 7.7 Merge

Allowed:

- Use the specifically approved merge method.
- Merge only the approved PR head.
- Permit remote branch deletion only when explained and approved.

Not allowed:

- Merging a changed head commit.
- Treating merge approval as deployment approval.
- Deleting local work without explicit approval.

Exit requirement:

- Merge is verified.
- Merge evidence is recorded.
- Deployment remains blocked until separately approved.

### 7.8 Deployment

Allowed only with explicit deployment approval:

- Deploy the exact approved artifact.
- Deploy only to the named environment.
- Perform approved verification.
- Record deployment evidence.
- Perform an approved rollback if required.

Not allowed:

- Automatic deployment merely because a merge succeeded.
- Deployment to an unspecified target.
- Expansion beyond the approved scope.
- Unapproved installation, communication, purchase, deletion, or
  protected-system modification.

## 8. Legal transitions

Normal transitions are:

- Research -> Specification
- Specification -> Prototype
- Prototype -> Tests
- Tests -> Security review
- Security review -> PR
- PR -> Merge
- Merge -> Deployment

Controlled remediation transitions are:

- Tests -> Prototype remediation
- Security review -> Prototype remediation
- PR review -> Prototype remediation
- Deployment verification -> approved rollback or remediation

After remediation, every downstream gate whose evidence may have been
invalidated must be repeated.

No other transition is valid unless Adrien explicitly approves a documented,
scoped exception and its consequences are explained.

## 9. Protected actions

The following require separate explicit authorization:

- Installation
- Dependency changes
- Purchases or financial commitments
- Emails, messages, notifications, or external communication
- File deletion
- Local branch deletion
- Remote branch deletion
- Protected-system modification
- Account modification
- Push
- Force-push
- Material external pull-request publication
- Merge
- Deployment
- Rollback
- Secret creation
- Secret rotation
- Secret transfer or disclosure
- Computer-use outside an approved sandbox
- Destructive database operations
- Irreversible migrations

A stage-transition approval does not automatically authorize a protected
action.

Adrien may approve multiple protected actions together only when every action,
target, side effect, and exclusion is clearly stated.

A direct yes may authorize one clearly presented protected-action request.

A direct no must deny the action.

<!-- GARVIS SPECIFICATION PART 2 COMPLETE -->

## 10. Core governance data model

### 10.1 Stage values

The universal stage-gate controller must represent these stages explicitly:

- research
- specification
- prototype
- tests
- security_review
- pr
- merge
- deployment
- prototype_remediation
- rollback_review

The controller must not rely only on free-form stage names.

### 10.2 Project record

Each governed project must have a project record containing:

- Unique project identifier
- Human-readable project name
- Creator attribution to Adrien D. Thomas
- System designation, including GARVIS AGI Beta when applicable
- Development track, including UPGRADE 2 when applicable
- Repository
- Worktree
- Branch
- Base commit
- Current artifact hash
- Approved file scope
- Current governance stage
- Completed stages
- Pending gate
- Current limitations
- Scientific AGI-validation status
- Created timestamp
- Updated timestamp
- Project status
- Previous audit-record hash

### 10.3 Approval question

A yes or no decision must be attached to one stored approval question.

The approval question must contain:

- Unique question identifier
- Project identifier
- Requested transition or protected action
- Plain-language explanation
- Exact target
- Approved file scope
- Repository
- Branch
- Commit or artifact hash
- Environment when applicable
- Expected valid answers
- UTC creation timestamp
- Expiration timestamp when applicable
- Previous audit-record hash
- Record hash

Changing the approval question must invalidate any answer previously attached
to it.

### 10.4 Transition approval

A transition approval must contain:

- Unique approval identifier
- Project identifier
- Approver name
- Source stage
- Destination stage
- Scope
- Repository
- Branch
- Commit or artifact hash
- Exact approval declaration
- Approval-question identifier when yes or no was used
- UTC timestamp
- Expiration when applicable
- One-time-use state
- Consumption timestamp
- Revocation state
- Revocation reason when applicable
- Previous audit-record hash
- Record hash

### 10.5 Protected-action authorization

A protected-action authorization must contain:

- Unique authorization identifier
- Project identifier
- Action type
- Exact target
- Repository, account, device, system, or environment
- Branch and commit when applicable
- Allowed side effects
- Prohibited side effects
- Reversibility statement
- Exact approval declaration
- Approval-question identifier when yes or no was used
- UTC timestamp
- Expiration
- One-time-use state
- Consumption state
- Revocation state
- Execution result
- Result evidence
- Previous audit-record hash
- Record hash

### 10.6 Scoped exception

A scoped exception must contain:

- Unique exception identifier
- Default restriction being overridden
- Reason for the exception
- Project
- Governance stage
- Exact allowed behavior
- Exact forbidden behavior
- Target
- File or data scope
- Repository
- Branch
- Commit or artifact hash
- Environment
- Start time
- Expiration or one-time-use condition
- Revocation state
- Exact approval declaration
- Previous audit-record hash
- Record hash

A scoped exception must not be valid outside its recorded scope.

### 10.7 Evidence record

An evidence record must contain:

- Unique evidence identifier
- Project identifier
- Governance stage
- Evidence type
- Artifact hash
- Command, workflow, review, or evaluation identifier
- Result
- UTC timestamp
- Environment summary
- Known limitations
- Previous evidence-record hash
- Record hash

Evidence for one artifact must not validate another artifact.

## 11. Tamper-evident audit chain

Every governance record must contain the hash of the previous record.

The new record hash must be calculated from a deterministic canonical
representation of that record, excluding its own record-hash field.

The audit system must detect:

- Modified records
- Deleted records
- Reordered records
- Duplicated one-time approvals
- Broken previous-record references
- Approval replay
- Approval for another project
- Approval for another stage
- Approval for another action
- Approval for another branch
- Approval for another commit
- Evidence for another artifact
- Expired authorization
- Revoked authorization
- Altered yes or no approval context

A broken or unverifiable chain must block the transition or protected action.

Audit verification must be read-only and must not repair damaged records
silently.

## 12. Approval validity and lifecycle

An approval is valid only when all required values match:

- Project
- Source stage
- Destination stage or protected action
- Scope
- Target
- Repository
- Branch
- Commit or artifact hash
- Environment
- Approver
- Approval-question identifier when applicable
- Approval-question text hash
- Expiration state
- One-time-use state
- Revocation state
- Audit-chain integrity

A material artifact change invalidates downstream test, security-review, PR,
merge, and deployment approvals.

A one-time approval must be marked consumed atomically with the protected
operation.

A consumed approval must not be reused.

An approval may be revoked before use.

A denial must not be converted into approval without a new explicit decision.

If GARVIS cannot prove that an approval is valid, GARVIS must deny the action
and explain which requirement failed.

## 13. Proposed governance interfaces

The later Prototype stage may implement interfaces conceptually equivalent to:

- create_project
- get_project_status
- request_transition
- create_approval_question
- record_direct_approval
- record_yes_no_decision
- approve_transition
- deny_transition
- authorize_protected_action
- deny_protected_action
- grant_scoped_exception
- revoke_authorization
- verify_transition
- verify_protected_action
- consume_authorization
- record_evidence
- verify_audit_chain
- render_plain_language_status
- render_capability_claim_status

The exact names may change during Prototype review.

Their default-deny and scope-binding security properties must not be removed.

## 14. Required plain-language status output

The status view must show:

- Project name
- GARVIS AGI Beta designation
- Scientific AGI-validation status
- UPGRADE 2 status
- Current governance stage
- Completed stages
- Current artifact hash
- Approved file scope
- Active transition approvals
- Active protected-action authorizations
- Active scoped exceptions
- Expired approvals
- Revoked approvals
- Consumed one-time approvals
- Latest test evidence
- Latest security-review evidence
- Actions GARVIS may perform
- Actions GARVIS must refuse
- Current limitations
- Exact pending approval
- Whether a direct yes or no is sufficient
- Practical consequences of the next action
- Whether unrelated local work is preserved
- Whether the action is reversible

## 15. Storage requirements

The first prototype should use an existing local deterministic storage option.

Preferred candidates are:

- SQLite from the Python standard library
- Canonical append-only JSON records

No new dependency is authorized by this specification.

Storage must:

- Be local by default
- Avoid secrets in audit logs
- Preserve UTC timestamps
- Support deterministic serialization
- Support transactions when applicable
- Prevent partial stage transitions
- Support one-time authorization consumption
- Support revocation
- Support expiration
- Support audit-chain verification
- Support safe read-only status queries
- Fail closed on corruption
- Preserve attribution to Adrien D. Thomas

The first Prototype should prefer SQLite unless repository inspection identifies
a stronger reason to use canonical JSON.

## 16. Integration boundaries

The stage-gate foundation must be reusable by:

- Reasoning systems
- Memory systems
- Visual capabilities
- Voice capabilities
- Document workflows
- Coding workflows
- Agent systems
- Simulations
- Dashboards
- Computer-use
- Network research
- Local file access
- UPGRADE 2 capabilities
- AGI Beta evaluations
- Future deployment systems

Existing capability-specific approval systems remain active.

The universal stage gate supplements them. It must not silently weaken,
replace, or bypass them.

## 17. Proposed Prototype file boundary

The later Prototype stage may propose these files:

- src/garvis/stage_gate.py
- src/garvis/stage_gate_store.py
- tests/garvis/test_stage_gate.py
- docs/GARVIS_STAGE_GATE_FOUNDATION.md

The exact file boundary requires Adrien's approval before implementation.

Existing capability broker, capability runtime, local-file access, memory, CI,
UPGRADE 2, and deployment files remain outside scope unless explicitly added.

<!-- GARVIS SPECIFICATION PART 3 COMPLETE -->

## 18. Required prototype tests

The future stage-gate prototype must prove the following behaviors.

### 18.1 Stage-transition enforcement

1. Each normal legal transition succeeds only with matching approval.
2. A project cannot skip a required stage.
3. A transition to an unknown stage is denied.
4. Approval for an earlier transition cannot authorize a later transition.
5. Approval for one project cannot authorize another project.
6. Approval for one branch cannot authorize another branch.
7. Approval for one commit cannot authorize a changed commit.
8. Approval for one artifact hash cannot authorize another artifact.
9. Missing transition approval causes denial.
10. Expired transition approval causes denial.
11. Revoked transition approval causes denial.
12. A denied transition remains denied until a new explicit decision exists.
13. Concurrent transition attempts cannot create contradictory project state.
14. Interrupted transitions do not leave partially updated project state.

### 18.2 Yes and no decision binding

15. A direct yes can approve one clearly identified approval question.
16. That yes cannot authorize a different approval question.
17. Changing the question text invalidates the stored answer.
18. Changing the action, target, branch, commit, or scope invalidates the answer.
19. An unclear response is not treated as approval.
20. Silence is not treated as approval.
21. A direct no denies the requested action.
22. A denial cannot be silently converted into approval.
23. One-time approval cannot be replayed.
24. A consumed approval cannot be used again.
25. Approval context is retained in the audit record.

### 18.3 Protected-action enforcement

26. Push requires push-specific authorization.
27. Push approval cannot authorize force-push.
28. Pull-request preparation cannot authorize merge.
29. Merge requires merge-specific authorization.
30. Merge approval cannot authorize deployment.
31. Remote branch deletion requires explicit inclusion in the approval.
32. Local branch deletion requires separate authorization.
33. Installation requires installation-specific authorization.
34. A dependency change requires explicit scope.
35. Purchases require exact financial and vendor scope.
36. Messages or external communications require exact recipient and purpose.
37. File deletion requires exact path scope.
38. Protected-system modification requires exact system and operation scope.
39. Secret creation, rotation, transfer, or disclosure requires exact approval.
40. Computer-use outside an approved sandbox is denied.
41. Destructive database operations are denied without specific authorization.
42. Irreversible migrations require explicit consequences and rollback review.

### 18.4 Scoped exceptions

43. A scoped exception works only inside its recorded project.
44. A scoped exception works only during its recorded stage.
45. A scoped exception works only for its exact target.
46. A scoped exception cannot expand its own file or data scope.
47. An expired exception is denied.
48. A revoked exception is denied.
49. A one-time exception cannot be replayed.
50. A narrow exception cannot become permanent blanket permission.
51. Forbidden side effects remain forbidden while the exception is active.

### 18.5 Evidence integrity

52. Test evidence must match the exact prototype artifact hash.
53. Security evidence must match the exact tested artifact hash.
54. A code change invalidates downstream evidence.
55. Prototype remediation requires affected tests to be repeated.
56. Security remediation requires affected tests and security review to repeat.
57. PR remediation invalidates approvals tied to the earlier head commit.
58. Evidence from another project is rejected.
59. Evidence from another environment is rejected when environment identity is
    material.

### 18.6 Audit-chain integrity

60. A modified record is detected.
61. A deleted record is detected.
62. A reordered record is detected.
63. A duplicated one-time authorization is detected.
64. An incorrect previous-record hash is detected.
65. A changed approval-question record is detected.
66. A changed yes or no decision record is detected.
67. A broken audit chain blocks the transition or protected action.
68. Audit verification does not silently repair damaged records.
69. Canonical serialization produces stable hashes.
70. Sensitive secrets are not written into audit records.

### 18.7 AGI Beta claim controls

71. GARVIS may identify itself as GARVIS AGI Beta.
72. GARVIS may state that full AGI is a development objective.
73. GARVIS may state that UPGRADE 2 advances that development objective.
74. GARVIS must disclose that scientific AGI validation is not established.
75. GARVIS must not claim scientifically proven AGI without approved evidence.
76. GARVIS must not independently promote itself from AGI Beta.
77. A future status elevation requires the approved validation process.
78. Capability claims remain limited to demonstrated evidence.
79. Known limitations remain visible in status output.
80. Future capability must not be presented as guaranteed.

### 18.8 Plain-language status and preservation

81. Status shows the current stage.
82. Status shows completed and pending gates.
83. Status shows the exact approval required next.
84. Status explains what will change and what will not change.
85. Status explains technical Git and GitHub terms in practical language.
86. Status identifies whether a direct yes or no is sufficient.
87. Status lists active, expired, revoked, and consumed approvals.
88. Status lists active scoped exceptions.
89. Status identifies protected actions that remain prohibited.
90. Unrelated local work remains outside the approved scope.
91. A worktree or branch mismatch causes safe refusal.
92. Deployment remains blocked after merge until separately approved.

## 19. Security-review requirements

The security review of the future prototype must evaluate:

- Default-deny behavior
- Legal-transition enforcement
- Approval-question identity and text binding
- Yes and no interpretation
- Approval scope binding
- Target binding
- Repository and branch binding
- Commit and artifact-hash binding
- Environment binding
- Expiration enforcement
- Revocation enforcement
- One-time-use enforcement
- Approval replay resistance
- Scoped-exception containment
- Audit-chain integrity
- Canonical record serialization
- Transaction safety
- Race conditions
- Interrupted-operation recovery
- Partial-write prevention
- Log injection
- Path validation
- Branch-name validation
- Command-construction safety
- Shell-injection resistance
- Secret handling
- Filesystem permissions
- Database permissions
- Protected-action isolation
- Capability-specific approval compatibility
- Push and force-push separation
- Merge and deployment separation
- Remote and local branch-deletion separation
- Installation and dependency-change isolation
- Purchase and communication authorization
- Destructive-operation protection
- Rollback authorization
- AGI Beta claim accuracy
- Scientific-validation claim protection
- Preservation of unrelated local work

Security review must inspect the exact artifact that passed the required tests.

Any material modification after testing or security review invalidates the
affected evidence and requires the appropriate downstream gates to be repeated.

A failed security review must not proceed directly to PR.

A failed review must return through an approved Prototype remediation path.

<!-- GARVIS SPECIFICATION PART 4A COMPLETE -->

## 20. Acceptance criteria

The GARVIS AGI Beta stage-gate foundation is acceptable only when all required
conditions below are satisfied.

### 20.1 Governance acceptance

- The official governance stages are explicitly represented.
- Every normal transition follows the approved sequence.
- Illegal or skipped transitions are denied.
- Every transition is bound to valid Adrien authorization.
- Every approval is bound to its exact project and scope.
- Every approval is bound to the correct repository and branch.
- Every approval is bound to the correct commit or artifact hash.
- Changed artifacts invalidate affected downstream approvals.
- Failed gates stop the project safely.
- Remediation returns through the correct earlier stages.
- Deployment remains independent from merge.

### 20.2 Adrien authorization acceptance

- Adrien D. Thomas remains the final project approval authority.
- A direct yes is accepted only for one clearly identified question.
- A direct no denies that exact request.
- Silence or unclear language is not treated as approval.
- Approval for one action cannot authorize another action.
- Approval for one project cannot authorize another project.
- One-time authorization cannot be replayed.
- Expired authorization is denied.
- Revoked authorization is denied.
- Scoped exceptions remain inside their exact limits.
- Technical consequences are explained before approval is requested.
- The status output identifies the exact decision required next.

### 20.3 Protected-action acceptance

- Push requires push-specific authorization.
- Force-push requires separate explicit authorization.
- Pull-request preparation does not authorize merge.
- Merge requires merge-specific authorization.
- Merge does not authorize deployment.
- Remote branch deletion is explicitly explained and approved.
- Local branch deletion is separately controlled.
- Installation and dependency changes require approval.
- Purchases require exact cost and vendor scope.
- External communication requires exact recipient and purpose.
- File deletion requires exact path scope.
- Protected-system changes require exact target and operation scope.
- Secret operations require explicit authorization.
- Destructive or irreversible actions require explicit consequences.
- Unrelated local work remains preserved.

### 20.4 Audit and evidence acceptance

- Governance records use deterministic serialization.
- Every record links to the previous record hash.
- Modified records are detectable.
- Deleted records are detectable.
- Reordered records are detectable.
- Replayed approvals are detectable.
- Approval-question changes invalidate old answers.
- Test evidence matches the exact tested artifact.
- Security evidence matches the exact reviewed artifact.
- Audit-chain corruption blocks the affected action.
- Audit verification does not silently repair evidence.
- Secrets are excluded from governance logs.
- Evidence records include known limitations.

### 20.5 AGI Beta acceptance

- GARVIS may identify itself as GARVIS AGI Beta.
- GARVIS may describe full AGI as a development objective.
- GARVIS may describe UPGRADE 2 as progress toward broader capability.
- GARVIS discloses that scientific AGI validation is not established.
- GARVIS does not claim scientifically proven AGI without approved evidence.
- GARVIS cannot independently elevate its scientific status.
- Capability statements remain proportional to demonstrated behavior.
- Known limitations remain visible.
- Future capability is not presented as guaranteed.
- Any future AGI-status elevation follows a separate validation process.

### 20.6 Engineering acceptance

- Required focused tests pass.
- Required full tests pass.
- Required denial cases pass.
- Required concurrency and transaction tests pass.
- Security review passes.
- CI passes.
- Pull-request findings are resolved.
- The reviewed PR head matches the approved commit.
- Merge is separately approved and verified.
- Deployment remains separately blocked.
- No unrelated files are included.

## 21. Non-goals for the first prototype

The first stage-gate prototype will not:

- Claim scientifically validated AGI.
- Guarantee that full AGI will be achieved.
- Deploy GARVIS.
- Add autonomous push, merge, or deployment.
- Create permanent blanket approval.
- Treat one yes as universal permission.
- Install new dependencies without approval.
- Purchase services or products.
- Send messages or external communications.
- Delete branches or files without approval.
- Provide unrestricted computer control.
- Replace GitHub branch protections.
- Replace existing capability-specific approval systems.
- Weaken local-file or network approval controls.
- Modify unrelated UPGRADE 2 work.
- Modify deployment systems.
- Make scientific claims unsupported by evidence.

## 22. Specification review questions for Adrien D. Thomas

Before approving Specification -> Prototype, Adrien should review these
decisions.

1. Is GARVIS AGI Beta the correct official development designation?
2. Is the distinction between AGI Beta and scientifically proven AGI clear?
3. Does UPGRADE 2 correctly describe the ongoing capability-development path?
4. Should a direct yes remain valid only for the immediately presented,
   specific approval question?
5. Should protected-action approvals default to one-time use?
6. Should transition approvals default to one-time use?
7. Should approvals expire automatically?
8. If approvals expire, what should the default duration be?
9. Should remote branch deletion always be listed separately?
10. Should local branch deletion always require separate approval?
11. Should emergency rollback use its own protected-action category?
12. Should an emergency rollback still require a direct Adrien decision when
    Adrien is available?
13. Should SQLite be used for the first local governance store?
14. Is the proposed Prototype file boundary acceptable?
15. Should the first Prototype remain limited to local-only operation?
16. Should status output include an abbreviated view and a detailed view?
17. Should every approval question include a clear reversible or irreversible
    label?
18. Should every approval display whether cost, communication, deletion,
    installation, merge, or deployment is involved?
19. Are additional protected-action categories required?
20. Are additional plain-language explanation fields required?

## 23. Proposed first Prototype boundary

The proposed first implementation boundary is:

- src/garvis/stage_gate.py
- src/garvis/stage_gate_store.py
- tests/garvis/test_stage_gate.py
- docs/GARVIS_STAGE_GATE_FOUNDATION.md

The first Prototype should:

- Use only Python standard-library dependencies.
- Prefer SQLite for local transactional storage.
- Operate locally by default.
- Implement default-deny behavior.
- Implement legal stage transitions.
- Implement approval-question binding.
- Implement yes and no decisions.
- Implement one-time approvals.
- Implement revocation and expiration.
- Implement protected-action authorization records.
- Implement scoped exceptions.
- Implement a tamper-evident audit chain.
- Implement plain-language project status.
- Implement AGI Beta claim-status reporting.
- Include focused automated tests.

The first Prototype must not integrate directly with push, merge, deployment,
purchases, messaging, installation, deletion, or protected-system execution.

Those integrations require later specifications and separate approvals.

## 24. Current governance gate

Current project:

GARVIS AGI Beta Stage-Gate Foundation

Current development track:

UPGRADE 2

Current governance stage:

Specification

Current scientific-validation status:

Full AGI is a development objective. Scientifically validated AGI has not been
established.

Current authorized work:

- Complete and inspect this specification document.
- Propose corrections.
- Prepare the exact Prototype boundary for approval.

Current unauthorized work:

- Implementing the stage-gate source modules
- Running Prototype-stage implementation tests
- Installing dependencies
- Pushing
- Opening or updating a pull request
- Merging
- Deleting files or branches
- Sending external communications
- Purchasing
- Modifying protected systems
- Deploying

Next transition:

Specification -> Prototype

Required next decision:

Adrien D. Thomas must review this completed specification and explicitly
approve or reject the proposed Prototype stage and file boundary.

No Prototype implementation is authorized by completion of this document.

<!-- GARVIS SPECIFICATION PART 4B COMPLETE -->
<!-- GARVIS AGI BETA STAGE-GATE SPECIFICATION DRAFT COMPLETE -->

## 25. Adrien D. Thomas review decisions

- Official designation: GARVIS AGI Beta.
- Full AGI is the development objective; scientific validation is not established.
- UPGRADE 2 is the controlled capability-development path.
- Yes or no applies only to the exact immediately presented approval question.
- Transition and protected-action approvals default to one-time use.
- Unused protected-action approvals expire after 10 minutes by default.
- Unused transition approvals expire after 24 hours by default.
- Completed audit records remain permanent historical evidence.
- Local and remote branch deletion must be separately identified.
- Rollback is a separate protected action; autonomous rollback is excluded.
- The first store will use local SQLite from Python's standard library.
- The first Prototype remains local-only.
- The approved Prototype boundary contains exactly four files.
- Status must provide both brief and detailed views.
- Approval prompts must show risk, reversibility, targets, and protected effects.

Approved Prototype boundary:

- src/garvis/stage_gate.py
- src/garvis/stage_gate_store.py
- tests/garvis/test_stage_gate.py
- docs/GARVIS_STAGE_GATE_FOUNDATION.md

Next approved stage:

Specification -> Prototype

Push, PR publication, merge, installation, deletion, communication, purchase,
protected-system modification, and deployment remain separately prohibited.

<!-- ADRIEN SPECIFICATION REVIEW COMPLETE -->
