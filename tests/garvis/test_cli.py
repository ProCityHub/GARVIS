from __future__ import annotations

import asyncio

import pytest

from garvis import cli


class FakeLocalRuntime:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def respond(self, message: str) -> str:
        self.messages.append(message)
        return f"local reply: {message}"


def test_cli_accepts_one_shot_prompt() -> None:
    args = cli.build_parser().parse_args(["What", "is", "GARVIS?"])

    assert args.prompt == ["What", "is", "GARVIS?"]
    assert args.remote is False
    assert args.session == "default"
    assert args.no_memory is False


def test_cli_supports_remote_memory_and_model_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "--remote",
            "--model",
            "test-model",
            "--session",
            "adrien",
            "--no-memory",
            "hello",
        ]
    )

    assert args.remote is True
    assert args.model == "test-model"
    assert args.session == "adrien"
    assert args.no_memory is True
    assert args.prompt == ["hello"]


def test_default_run_uses_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeLocalRuntime()
    monkeypatch.setattr(cli, "_build_local_runtime", lambda: runtime)

    args = cli.build_parser().parse_args(["hello"])
    result = asyncio.run(cli._run(args))

    assert result == 0
    assert runtime.messages == ["hello"]
    assert capsys.readouterr().out.strip() == "local reply: hello"


class FakeApprovalRuntime:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def respond(self, message: str) -> str:
        self.messages.append(message)
        if len(self.messages) == 1:
            return "GARVIS requests internet research permission.\n\nApprove? [Y/N]"
        return "Approved research answer."


def test_local_one_shot_consumes_approval(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeApprovalRuntime()
    monkeypatch.setattr(cli, "_build_local_runtime", lambda: runtime)
    monkeypatch.setattr("builtins.input", lambda: "y")

    args = cli.build_parser().parse_args(["research", "current", "drywall", "prices"])
    result = asyncio.run(cli._run(args))

    assert result == 0
    assert runtime.messages == [
        "research current drywall prices",
        "y",
    ]

    output = capsys.readouterr().out
    assert "Approve? [Y/N]" in output
    assert output.rstrip().endswith("Approved research answer.")
