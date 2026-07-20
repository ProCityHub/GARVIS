import asyncio
import json
from pathlib import Path

import pytest

from garvis.cli import _run, build_parser
from garvis.lattice_cycle_cli import load_evidence_envelope


def write_strong_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "immutable_source_evidence": {
                    "ultimatum-result": {
                        "status": "negative",
                        "auc_phi": 0.871258,
                        "auc_flat": 0.861944,
                    }
                },
                "deterministic_agi_measurements": {
                    "difference": 0.009314,
                },
            }
        ),
        encoding="utf-8",
    )


def test_parser_accepts_local_lattice_cycle(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"

    args = build_parser().parse_args(
        [
            "--lattice-cycle",
            str(evidence_path),
            "--cycle",
            "7",
            "--external-action",
        ]
    )

    assert args.lattice_cycle == evidence_path
    assert args.cycle == 7
    assert args.external_action is True


def test_local_cycle_does_not_require_openai_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence_path = tmp_path / "evidence.json"
    write_strong_evidence(evidence_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = build_parser().parse_args(
        [
            "--lattice-cycle",
            str(evidence_path),
            "--cycle",
            "7",
            "--external-action",
        ]
    )

    code = asyncio.run(_run(args))
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert code == 0
    assert captured.err == ""
    assert output["mode"] == "local_lattice_cycle"
    assert output["cycle"] == 7
    assert output["pulse"]["raw_union"] == pytest.approx(1.6)
    assert output["pulse"]["normalized_center"] == (
        pytest.approx(1.0)
    )
    assert output["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert output["human_approval_required"] is True
    assert output["external_action_allowed"] is False


def test_lattice_cycle_rejects_conversation_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence_path = tmp_path / "evidence.json"
    write_strong_evidence(evidence_path)

    args = build_parser().parse_args(
        [
            "--lattice-cycle",
            str(evidence_path),
            "answer",
            "this",
        ]
    )

    code = asyncio.run(_run(args))
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert "cannot be combined" in captured.err


def test_invalid_json_returns_clear_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence_path = tmp_path / "invalid.json"
    evidence_path.write_text("{not-json", encoding="utf-8")

    args = build_parser().parse_args(
        ["--lattice-cycle", str(evidence_path)]
    )

    code = asyncio.run(_run(args))
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert "not valid JSON" in captured.err


def test_unknown_evidence_fields_are_rejected(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "unknown.json"
    evidence_path.write_text(
        json.dumps(
            {
                "immutable_source_evidence": {
                    "result": "observed"
                },
                "unregistered_field": {
                    "value": 1
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsupported evidence fields",
    ):
        load_evidence_envelope(evidence_path)


def test_local_adapter_has_no_execution_interface(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    write_strong_evidence(evidence_path)

    envelope = load_evidence_envelope(evidence_path)

    assert not hasattr(envelope, "execute")
    assert not hasattr(envelope, "send")
    assert not hasattr(envelope, "connect")
