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


# CANONICAL RESEARCH MEMORY RUNTIME TESTS

def test_epistemic_request_requires_research_memory_context() -> None:
    from garvis.local_language_runtime import (
        _recall_runtime_memory_context,
        _research_memory_required,
        classify_request,
    )

    class FakeMemory:
        def __init__(self) -> None:
            self.research_calls = 0
            self.general_calls = 0

        def render_research_context(
            self,
            query: str,
            *,
            session_id: str,
        ) -> str:
            self.research_calls += 1
            return (
                "[research-memory-control consulted=true "
                "execution_authority=false] "
                + query
                + " "
                + session_id
            )

        def render_context(
            self,
            query: str,
            *,
            session_id: str,
        ) -> str:
            self.general_calls += 1
            return "GENERAL " + query + " " + session_id

    envelope = classify_request(
        "research procedural memory and evidence boundaries"
    )

    required = _research_memory_required(envelope, "")
    fake = FakeMemory()

    context = _recall_runtime_memory_context(
        fake,
        envelope,
        session_id="research-test",
        research_memory_required=required,
    )

    assert required is True
    assert "consulted=true" in context
    assert "execution_authority=false" in context
    assert fake.research_calls == 1
    assert fake.general_calls == 0


def test_external_research_context_forces_research_memory_mode() -> None:
    from garvis.local_language_runtime import (
        _research_memory_required,
        classify_request,
    )

    envelope = classify_request(
        "What are current drywall prices?"
    )

    assert envelope.destination != "epistemic_registry"
    assert (
        _research_memory_required(
            envelope,
            "external internet evidence",
        )
        is True
    )



def test_recent_release_question_requires_research_memory() -> None:
    from garvis.local_language_runtime import (
        _research_memory_required,
        classify_request,
    )

    envelope = classify_request(
        "Find out whether Python 3.14 has been released"
    )

    assert _research_memory_required(envelope, "") is True
