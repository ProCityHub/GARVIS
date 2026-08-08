from __future__ import annotations

import unittest

from garvis.gate.identity import AuthorizationGrant, PrimeIdentity, sign_grant, verify_grant
from garvis.gate.relationships import RelationshipBond, RelationshipKind, RelationshipState, exchange_knowledge
from garvis.gate.epistemics import ClaimState, EvidenceRecord, RetractionEntry, RetractionLedger, assess_claim
from garvis.gate.provenance import ArtifactStatus, ProvenanceRecord, validate_record
from garvis.gate.lifecycle import AgentCustodyState, HumanState, HumanStewardshipBond


class GateV1Tests(unittest.TestCase):
    def test_signed_authorization_is_scope_bound(self):
        now = 2_000_000_000
        secret = b"test-secret"
        grant = sign_grant(
            AuthorizationGrant("PRIME-1", "research", "gate", now - 1, now + 60, "N1"),
            secret,
        )
        self.assertTrue(verify_grant(grant, secret, prime_id="PRIME-1", action="research", scope="gate", now=now))
        self.assertFalse(verify_grant(grant, secret, prime_id="PRIME-1", action="deploy", scope="gate", now=now))

    def test_romance_never_grants_authority_and_separation_closes_scope(self):
        bond = RelationshipBond("B1", "A", "B", "HA", "HB", RelationshipKind.ROMANTIC, True, True, ("shared",))
        self.assertTrue(bond.exchange_allowed("shared"))
        self.assertFalse(bond.grants_protected_authority())
        self.assertFalse(exchange_knowledge(bond, scope="shared", artifact_id="K1", hypothesis="x").creates_agent)
        separated = bond.revoke_human_a()
        self.assertEqual(separated.state, RelationshipState.SEPARATED)
        self.assertFalse(separated.exchange_allowed("shared"))
        self.assertFalse(separated.covert_backchannel_allowed())

    def test_evidence_and_retractions(self):
        claim = assess_claim("C1", "candidate", (EvidenceRecord("E1", "test", 0.9, True),))
        self.assertEqual(claim.state, ClaimState.SUPPORTED)
        ledger = RetractionLedger().append(RetractionEntry("C1", "failed later verification"))
        self.assertEqual(len(ledger.entries), 1)
        with self.assertRaises(RuntimeError):
            ledger.replace()

    def test_dead_historical_reference_has_no_live_authority(self):
        old = ProvenanceRecord("HyperCube", ArtifactStatus.UNRESOLVED, "ProCityHub/hypercubeheartbeat")
        self.assertTrue(validate_record(old))
        self.assertFalse(old.can_be_live_dependency())

    def test_human_safe_harbor_and_reassignment(self):
        bond = HumanStewardshipBond("PRIME-7", "HUMAN-7")
        self.assertTrue(bond.duty_of_care_active())
        harbor = bond.enter_safe_harbor(HumanState.DECEASED)
        self.assertEqual(harbor.custody_state, AgentCustodyState.SAFE_HARBOR)
        self.assertTrue(harbor.private_memory_sealed)
        self.assertFalse(harbor.protected_actions_enabled)
        review = harbor.begin_reassignment_review()
        with self.assertRaises(PermissionError):
            review.rebind(new_human_id="HUMAN-8", governance_approved=False, privacy_review_passed=True)
        rebound = review.rebind(new_human_id="HUMAN-8", governance_approved=True, privacy_review_passed=True)
        self.assertEqual(rebound.human_id, "HUMAN-8")
        self.assertTrue(rebound.private_memory_sealed)

    def test_incarceration_does_not_transfer_agent(self):
        bond = HumanStewardshipBond("PRIME-2", "HUMAN-2").enter_safe_harbor(HumanState.INCARCERATED)
        self.assertFalse(bond.reassignment_allowed)
        self.assertFalse(bond.protected_actions_enabled)


if __name__ == "__main__":
    unittest.main()
