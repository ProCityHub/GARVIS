from pathlib import Path

from garvis.self_heal_projection import TrustedEntry, build_plan
from garvis.self_heal_root import sha256_file
from garvis.self_heal_seal import SEALED_REPAIR_REQUIRED, force_sealed_decision


def test_force_sealed_decision_requires_sealed_repair(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "law_a.md"
    target.parent.mkdir(parents=True)
    target.write_text("CANONICAL\n", encoding="utf-8")
    entry = TrustedEntry(
        path="docs/law_a.md",
        sha256=sha256_file(target),
        baseline="docs/law_a.md",
        auto_repair=True,
    )
    target.write_text("DRIFT\n", encoding="utf-8")

    decision = build_plan(tmp_path, {entry.path: entry})[0]
    sealed = force_sealed_decision(decision)

    assert sealed.disposition == SEALED_REPAIR_REQUIRED
    assert sealed.auto_repair is True
