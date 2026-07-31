from __future__ import annotations

import unittest

from garvis.local_language_runtime import FilingEnvelope, render_local_prompt


class LocalRuntimeMemoryPromptTests(unittest.TestCase):
    def test_prompt_contains_memory_context(self) -> None:
        envelope = FilingEnvelope(
            destination="engineering_registry",
            evidence_status="user_supplied",
            authority="adrien_user_input",
            permission="local_response_only",
            request="Continue the local runtime",
        )
        prompt = render_local_prompt(
            envelope,
            "[memory id=3 evidence=user_supplied] Use local GGUF.",
        )
        self.assertNotIn("GARVIS_MEMORY_CONTEXT_BEGIN", prompt)
        self.assertIn("Use this fallible recalled context only when relevant:", prompt)
        self.assertIn("[memory id=3 evidence=user_supplied] Use local GGUF.", prompt)
        self.assertIn("Use local GGUF.", prompt)
        self.assertNotIn("GARVIS_MEMORY_CONTEXT_END", prompt)

    def test_empty_context_is_omitted(self) -> None:
        envelope = FilingEnvelope(
            destination="general_dialogue",
            evidence_status="user_supplied",
            authority="adrien_user_input",
            permission="local_response_only",
            request="Hello",
        )
        self.assertNotIn(
            "GARVIS_MEMORY_CONTEXT_BEGIN",
            render_local_prompt(envelope),
        )


if __name__ == "__main__":
    unittest.main()
