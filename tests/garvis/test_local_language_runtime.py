from __future__ import annotations

import unittest

from garvis.local_language_runtime import (
    FilingEnvelope,
    classify_request,
    clean_model_output,
    render_local_prompt,
)


class LocalLanguageRuntimeTests(unittest.TestCase):
    def test_routes_build_request(self) -> None:
        envelope = classify_request("Build the local GARVIS runtime")
        self.assertEqual(envelope.destination, "engineering_registry")
        self.assertEqual(envelope.permission, "local_response_only")

    def test_preserves_speculation_as_provisional(self) -> None:
        envelope = classify_request("Maybe this hypothesis is scientifically useful")
        self.assertEqual(envelope.destination, "epistemic_registry")
        self.assertEqual(envelope.evidence_status, "provisional_claim")

    def test_external_action_requires_approval(self) -> None:
        envelope = classify_request("Please publish this report")
        self.assertEqual(
            envelope.permission,
            "approval_required_before_external_action",
        )

    def test_prompt_contains_filing_and_no_think(self) -> None:
        envelope = FilingEnvelope(
            destination="engineering_registry",
            evidence_status="user_supplied",
            authority="adrien_user_input",
            permission="local_response_only",
            request="Test request",
        )
        prompt = render_local_prompt(envelope)
        self.assertTrue(prompt.startswith("/no_think "))
        self.assertNotIn("GARVIS_FILING_ENVELOPE=", prompt)
        self.assertIn("Operate with local response only permission", prompt)
        self.assertIn('User request: "Test request"', prompt)
        self.assertIn("focus on engineering registry", prompt)

    def test_prompt_keeps_evidence_channels_separate(self) -> None:
        envelope = FilingEnvelope(
            destination="engineering_registry",
            evidence_status="user_supplied",
            authority="adrien_user_input",
            permission="local_response_only",
            request="Explain the code",
        )
        prompt = render_local_prompt(
            envelope,
            repository_context="--- src/garvis/cli.py ---\nACTIVE_CLI",
            external_context="PUBLIC INTERNET RESEARCH CONTEXT",
            workspace_context="APPROVED READ-ONLY LOCAL FILE EVIDENCE",
        )
        self.assertIn("read-only local repository evidence", prompt)
        self.assertIn("external internet evidence", prompt)
        self.assertIn("one-task approved local file evidence", prompt)

    def test_clean_output_removes_thinking(self) -> None:
        self.assertEqual(
            clean_model_output("<think>private reasoning</think>\nFINAL LOCAL ANSWER"),
            "FINAL LOCAL ANSWER",
        )


if __name__ == "__main__":
    unittest.main()
