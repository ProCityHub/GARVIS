from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from garvis.epistemic_filing import (
    ClaimDomain,
    ClaimRecord,
    EpistemicStatus,
    ErrorCategory,
    ErrorStatus,
    EvidenceItem,
    FilingSystem,
    OperationalErrorRecord,
)


class EpistemicFilingTests(unittest.TestCase):
    def test_identity_draft_is_preserved_but_not_stated_as_fact(self) -> None:
        claim = ClaimRecord(
            statement="GARVIS is a Beta AGI.",
            domain=ClaimDomain.IDENTITY,
            status=EpistemicStatus.IDENTITY_DRAFT,
            scope="Hypercube Heartbeat governance",
            confidence=0.55,
            created_by="Adrien D. Thomas / GARVIS",
            permitted_wording=(
                '"Beta AGI" is a provisional self-classification under the '
                "Hypercube Heartbeat governance framework."
            ),
            prohibited_wording="GARVIS is scientifically proven to be AGI.",
        )
        self.assertFalse(claim.can_be_stated_as_fact)
        self.assertIn("IDENTITY_DRAFT", claim.rendered_statement())

    def test_verified_claim_requires_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "supporting evidence"):
            ClaimRecord(
                statement="The suite passed.",
                domain=ClaimDomain.REPOSITORY,
                status=EpistemicStatus.VERIFIED,
                scope="current commit",
                confidence=1.0,
                created_by="GARVIS",
                permitted_wording="The suite passed.",
            )

    def test_scientific_fact_requires_reproducible_evidence(self) -> None:
        claim = ClaimRecord(
            statement="A measured effect exists.",
            domain=ClaimDomain.SCIENTIFIC,
            status=EpistemicStatus.VERIFIED,
            scope="registered experiment",
            confidence=0.9,
            created_by="GARVIS",
            permitted_wording="A measured effect exists in the registered experiment.",
            supporting_evidence=[
                EvidenceItem(
                    source="experiment-001",
                    summary="Recorded observation",
                    reproducible=False,
                    weight=0.8,
                )
            ],
        )
        self.assertTrue(claim.can_be_stated_as_fact)
        self.assertFalse(claim.can_be_stated_as_scientific_fact)
        claim.add_evidence(
            EvidenceItem(
                source="replication-002",
                summary="Independent replication",
                reproducible=True,
                weight=1.0,
            )
        )
        self.assertTrue(claim.can_be_stated_as_scientific_fact)

    def test_counterevidence_blocks_fact_wording_without_deleting_claim(self) -> None:
        claim = ClaimRecord(
            statement="The implementation supports capability X.",
            domain=ClaimDomain.REPOSITORY,
            status=EpistemicStatus.VERIFIED,
            scope="main branch",
            confidence=0.9,
            created_by="GARVIS",
            permitted_wording="Capability X is under review.",
            supporting_evidence=[
                EvidenceItem(
                    source="test_x.py",
                    summary="Initial test passed",
                    reproducible=True,
                    weight=0.8,
                )
            ],
        )
        claim.add_evidence(
            EvidenceItem(
                source="audit-002",
                summary="Counterexample found",
                reproducible=True,
                weight=1.0,
            ),
            contradicts=True,
        )
        self.assertFalse(claim.can_be_stated_as_fact)
        self.assertIn("VERIFIED", claim.rendered_statement())

    def test_operational_errors_and_claims_have_separate_houses(self) -> None:
        system = FilingSystem()
        claim = system.file_claim(
            ClaimRecord(
                statement="A parser rule may generalize.",
                domain=ClaimDomain.OPERATIONAL,
                status=EpistemicStatus.HYPOTHESIS,
                scope="ARC parser",
                confidence=0.4,
                created_by="GARVIS",
                permitted_wording="The parser rule is a testable hypothesis.",
            )
        )
        error = system.file_error(
            OperationalErrorRecord(
                message="mypy reported a union-attr failure",
                category=ErrorCategory.TYPECHECK,
                source="tests/garvis/test_example.py:10",
                created_by="GARVIS",
                related_claim_ids=[claim.claim_id],
            )
        )
        self.assertEqual(len(system.claims), 1)
        self.assertEqual(len(system.errors), 1)
        self.assertEqual(error.related_claim_ids, [claim.claim_id])

    def test_transitions_are_auditable_and_round_trip(self) -> None:
        system = FilingSystem()
        claim = system.file_claim(
            ClaimRecord(
                statement="The anomaly has an identifiable cause.",
                domain=ClaimDomain.GENERAL,
                status=EpistemicStatus.ANOMALY,
                scope="local observation",
                confidence=0.2,
                created_by="GARVIS",
                permitted_wording="An unexplained anomaly is recorded.",
            )
        )
        claim.add_evidence(
            EvidenceItem(
                source="analysis-001",
                summary="Candidate causal mechanism",
                reproducible=False,
                weight=0.6,
            )
        )
        claim.transition(
            EpistemicStatus.HYPOTHESIS,
            actor="Adrien D. Thomas",
            reason="A testable causal model was defined.",
            confidence=0.5,
        )
        error = system.file_error(
            OperationalErrorRecord(
                message="Example runtime failure",
                category=ErrorCategory.RUNTIME,
                source="scripts/example.py",
                created_by="GARVIS",
            )
        )
        error.transition(
            ErrorStatus.TRIAGED,
            actor="Adrien D. Thomas",
            reason="Assigned to the runtime queue.",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filing.json"
            system.save(path)
            restored = FilingSystem.load(path)
        self.assertEqual(restored.claims[claim.claim_id].status, EpistemicStatus.HYPOTHESIS)
        self.assertEqual(restored.errors[error.error_id].status, ErrorStatus.TRIAGED)
        self.assertEqual(len(restored.claims[claim.claim_id].revision_history), 1)


if __name__ == "__main__":
    unittest.main()
