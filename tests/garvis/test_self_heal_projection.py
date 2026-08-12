from pathlib import Path

from garvis.self_heal_projection import REPAIR_REQUIRED, REPORT_ONLY, TRUSTED, TrustedEntry, build_plan
from garvis.self_heal_root import sha256_file



def test_build_plan_marks_trusted_file(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "law_a.md"
    target.parent.mkdir(parents=True)
    target.write_text("LAW\n", encoding="utf-8")
    entry = TrustedEntry(
        path="docs/law_a.md",
        sha256=sha256_file(target),
        baseline="docs/law_a.md",
        auto_repair=True,
    )

    plan = build_plan(tmp_path, {entry.path: entry})

    assert plan[0].disposition == TRUSTED
    assert plan[0].observation.drifted is False



def test_build_plan_marks_drift_for_auto_repair(tmp_path: Path) -> None:
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

    plan = build_plan(tmp_path, {entry.path: entry})

    assert plan[0].disposition == REPAIR_REQUIRED
    assert plan[0].observation.drifted is True



def test_build_plan_marks_report_only_without_auto_repair(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "law_a.md"
    target.parent.mkdir(parents=True)
    target.write_text("CANONICAL\n", encoding="utf-8")
    entry = TrustedEntry(
        path="docs/law_a.md",
        sha256=sha256_file(target),
        baseline="docs/law_a.md",
        auto_repair=False,
    )
    target.write_text("DRIFT\n", encoding="utf-8")

    plan = build_plan(tmp_path, {entry.path: entry})

    assert plan[0].disposition == REPORT_ONLY
