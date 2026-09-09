# ProCityHub / GARVIS Architecture

Status: HYPOTHESIS_UNDER_TEST

This document defines the target native ProCityHub architecture. It does not claim that the current repository has completed this migration.

## Canonical domains

GARVIS - product coordination and human-facing interface.

Genesis X - project-defined cognitive processing architecture and heartbeat.

OAB - Observer Actor Bridge connecting observation, bounded action, and termination.

Hypercube - project mathematical, resonance, lattice, and heartbeat research layer.

Evidence - observations, frozen predictions, outcomes, contradictions, and provenance.

Governance - authority, approvals, permissions, policy, and audit.

Capabilities - bounded registry and execution of technically available actions.

Inference - project-controlled protocol boundary for inference implementations.

Interfaces - CLI, Android, voice, API, and other approved human-facing surfaces.

## Authority boundary

No architectural layer silently inherits another layer's authority.

Inference output is candidate material, not truth or governance authority.

Capability does not create authorization.

CAPABILITY IS NOT AUTHORIZATION.

## Genesis X heartbeat

RECEIVE -> SEGMENT -> PREDICT -> VERIFY -> SIMULATE -> PLAN -> OUTPUT -> FEEDBACK -> CONSOLIDATE

## Evidence witness

Raw_Pre -> Frozen_Prediction -> Change_or_Action -> Raw_Post -> Observed_Diff -> Contradiction_Error -> Consolidation

Predictions must not be rewritten after the outcome merely to make the earlier decision appear correct.

## Target package boundary

src/procityhub/garvis/
src/procityhub/genesis_x/
src/procityhub/oab/
src/procityhub/hypercube/
src/procityhub/evidence/
src/procityhub/governance/
src/procityhub/capabilities/
src/procityhub/inference/
src/procityhub/interfaces/

## Legacy migration boundary

The current repository still contains legacy external-framework and provider coupling. Useful ProCityHub-owned behavior must be migrated or rewritten before obsolete active paths are removed.

Historical attribution and applicable licence obligations must remain recoverable.

Replacement precedes removal.

## Research boundary

Software tests do not establish AGI, consciousness, ontology, or new physics.

Benchmark results are empirical evidence only and must remain separate from ontological interpretation.
