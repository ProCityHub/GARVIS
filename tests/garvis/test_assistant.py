from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from garvis.assistant import (
    ApprovalRequirement,
    GarvisAssistant,
    GarvisResponseError,
    assess_request,
)


@dataclass
class FakeResult:
    final_output: object


class FakeRunner:
    def __init__(self, output: object = "A direct GARVIS answer.") -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, agent: Any, input_text: str, **kwargs: Any) -> FakeResult:
        self.calls.append({"agent": agent, "input": input_text, **kwargs})
        return FakeResult(self.output)


def test_default_agent_has_no_external_tools() -> None:
    assistant = GarvisAssistant(persist_memory=False)

    assert assistant.agent.tools == []


def test_normal_question_is_not_approval_gated() -> None:
    assessment = assess_request("What is the purpose of the heartbeat?")

    assert assessment.approval_requirement is ApprovalRequirement.NONE
    assert assessment.requires_approval is False


def test_how_to_question_about_destructive_action_is_still_informational() -> None:
    assessment = assess_request("How do I delete an old GitHub branch safely?")

    assert assessment.requires_approval is False


@pytest.mark.parametrize(
    "message",
    [
        "Send this email to the client",
        "Could you publish the report?",
        "Go ahead and delete the remote branch",
        "Transfer the payment now",
    ],
)
def test_explicit_external_actions_require_approval(message: str) -> None:
    assessment = assess_request(message)

    assert assessment.requires_approval is True
    assert assessment.reason


@pytest.mark.asyncio
async def test_assistant_returns_direct_model_output() -> None:
    runner = FakeRunner("The heartbeat is operating normally.")
    assistant = GarvisAssistant(runner=runner, persist_memory=False)

    reply = await assistant.respond("Is the heartbeat operating?")

    assert reply.text == "The heartbeat is operating normally."
    assert reply.requires_approval is False
    assert runner.calls[0]["input"] == "Is the heartbeat operating?"
    assert runner.calls[0]["session"] is None


@pytest.mark.asyncio
async def test_external_action_is_prepared_without_claiming_execution() -> None:
    runner = FakeRunner("I prepared the email. Approval is required before sending it.")
    assistant = GarvisAssistant(runner=runner, persist_memory=False)

    reply = await assistant.respond("Send this email to the client")

    assert reply.requires_approval is True
    assert reply.approval_reason
    assert "outside-world action" in runner.calls[0]["input"]


@pytest.mark.asyncio
async def test_session_factory_is_reused_for_same_session(tmp_path: Path) -> None:
    runner = FakeRunner()
    created_sessions: list[Any] = []

    def session_factory(session_id: str, db_path: Path) -> object:
        session = {"session_id": session_id, "db_path": db_path}
        created_sessions.append(session)
        return session

    assistant = GarvisAssistant(
        runner=runner,
        session_db=tmp_path / "sessions.db",
        session_factory=session_factory,
    )

    await assistant.respond("First question", session_id="adrien")
    await assistant.respond("Second question", session_id="adrien")

    assert len(created_sessions) == 1
    assert runner.calls[0]["session"] is runner.calls[1]["session"]


@pytest.mark.asyncio
async def test_empty_model_output_raises_clear_error() -> None:
    assistant = GarvisAssistant(runner=FakeRunner(""), persist_memory=False)

    with pytest.raises(GarvisResponseError, match="without a text answer"):
        await assistant.respond("Answer this")


@pytest.mark.asyncio
async def test_empty_user_message_is_rejected() -> None:
    assistant = GarvisAssistant(runner=FakeRunner(), persist_memory=False)

    with pytest.raises(ValueError, match="message must not be empty"):
        await assistant.respond("   ")


# REMOTE CANONICAL RESEARCH MEMORY TESTS

@pytest.mark.asyncio
async def test_remote_research_memory_is_transient_even_without_session_persistence() -> None:
    sentinel = "REMOTE_RESEARCH_MEMORY_SENTINEL_817"
    provider_calls: list[tuple[str, str]] = []

    def memory_provider(query: str, session_id: str) -> str:
        provider_calls.append((query, session_id))
        return sentinel

    runner = FakeRunner("research response")

    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        research_memory_provider=memory_provider,
    )

    prompt = "Research current Python release changes"

    reply = await assistant.respond(
        prompt,
        session_id="remote-memory-test",
    )

    assert reply.text == "research response"
    assert provider_calls == [
        (prompt, "remote-memory-test")
    ]

    call = runner.calls[0]

    assert call["session"] is None
    assert call["input"] == prompt
    assert sentinel not in call["input"]

    transient_agent = call["agent"]

    assert transient_agent is not assistant.agent
    assert sentinel in str(transient_agent.instructions)
    assert "ADVISORY CONTEXT ONLY" in str(
        transient_agent.instructions
    )
    assert "not retrieved source evidence" in str(
        transient_agent.instructions
    ).lower()

    assert sentinel not in str(assistant.agent.instructions)


@pytest.mark.asyncio
async def test_remote_research_memory_failure_blocks_provider_inference() -> None:
    runner = FakeRunner("must not execute")

    def unavailable_memory(
        query: str,
        session_id: str,
    ) -> str:
        del query, session_id
        raise RuntimeError("memory unavailable")

    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        research_memory_provider=unavailable_memory,
    )

    with pytest.raises(
        GarvisResponseError,
        match="mandatory research memory unavailable",
    ):
        await assistant.respond(
            "Research current lattice evidence",
            session_id="remote-memory-test",
        )

    assert runner.calls == []


@pytest.mark.asyncio
async def test_nonresearch_remote_reasoning_does_not_require_research_memory() -> None:
    runner = FakeRunner("ordinary answer")
    provider_calls: list[str] = []

    def memory_provider(
        query: str,
        session_id: str,
    ) -> str:
        provider_calls.append(query + session_id)
        return "should not be used"

    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        research_memory_provider=memory_provider,
    )

    await assistant.respond(
        "Explain how drywall compound cures",
        session_id="ordinary",
    )

    assert provider_calls == []
    assert runner.calls[0]["agent"] is assistant.agent



@pytest.mark.asyncio
async def test_remote_primary_evidence_phrase_requires_research_memory() -> None:
    calls: list[str] = []

    def provider(query: str, session_id: str) -> str:
        calls.append(query)
        return "ADVISORY_MEMORY_CONTEXT"

    runner = FakeRunner("research answer")

    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        research_memory_provider=provider,
    )

    prompt = "Find recent primary evidence about memory consolidation"

    await assistant.respond(
        prompt,
        session_id="classifier-regression",
    )

    assert calls == [prompt]
