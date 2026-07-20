from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from garvis.resilient_runtime import (
    ResilientGarvisRuntime,
    split_wrapped_prompt,
)


class FakeLedger:
    def __init__(self) -> None:
        self.turns: List[Dict[str, str]] = []
        self.events: List[str] = []

    def append(self, role: str, content: str) -> None:
        self.events.append(f"append:{role}")
        self.turns.append({"role": role, "content": content})


class CapturingModel:
    def __init__(self, reply: str = "test reply") -> None:
        self.reply = reply
        self.messages: List[Dict[str, str]] = []

    def __call__(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, str]],
    ) -> str:
        del client, model
        self.messages = list(messages)
        return self.reply


def fake_build_context(
    system_prompt: str,
    ledger: FakeLedger,
) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *ledger.turns[-30:]]


def test_split_wrapped_prompt_removes_repeated_wrapper() -> None:
    context, message = split_wrapped_prompt(
        "constitutional controls\n\n"
        "ADRIEN'S CURRENT MESSAGE:\n"
        "remember this line"
    )

    assert context == "constitutional controls"
    assert message == "remember this line"


@pytest.mark.asyncio
async def test_user_is_persisted_before_model_and_reply_before_return() -> None:
    ledger = FakeLedger()
    model = CapturingModel("persisted reply")

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
    )

    reply = await runtime.respond("hello")

    assert reply.text == "persisted reply"
    assert ledger.events == ["append:user", "append:assistant"]
    assert ledger.turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "persisted reply"},
    ]


@pytest.mark.asyncio
async def test_wrapper_context_is_not_written_into_ledger() -> None:
    ledger = FakeLedger()
    model = CapturingModel()

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
    )

    await runtime.respond(
        "large constitutional wrapper\n\n"
        "ADRIEN'S CURRENT MESSAGE:\n"
        "small current message"
    )

    assert ledger.turns[0] == {
        "role": "user",
        "content": "small current message",
    }
    assert "large constitutional wrapper" in model.messages[0]["content"]
    assert "large constitutional wrapper" not in str(ledger.turns)


@pytest.mark.asyncio
async def test_failed_model_call_preserves_user_input() -> None:
    ledger = FakeLedger()

    def failing_model(
        client: Any,
        model: str,
        messages: List[Dict[str, str]],
    ) -> str:
        del client, model, messages
        raise RuntimeError("simulated 429")

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=failing_model,
    )

    with pytest.raises(RuntimeError, match="simulated 429"):
        await runtime.respond("survive the crash")

    assert ledger.turns == [
        {"role": "user", "content": "survive the crash"},
    ]
