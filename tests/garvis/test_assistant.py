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
