import os
from pathlib import Path

import pytest

from garvis.self_heal_executor import (
    RepairRefused,
    VerificationEvidence,
    expected_evidence_sha256,
    sealed_auto_repair,
)
from garvis.self_heal_projection import TrustedEntry, build_plan
from garvis.self_heal_root import build_canonical_root, compute_bundle, sha256_file
from garvis.self_heal_seal import force_sealed_decision


def build_environment(
    tmp_path: Path,
    *,
    target: str = "docs/law_a.md",
) -> tuple[Path, object, str]:
    root = tmp_path / "runtime"
    canonical = tmp_path / "canonical"
    baseline = root / ".garvis" / "baseline"
    root.mkdir()
    canonical.mkdir()
    baseline.mkdir(parents=True)

    authority_target = canonical / "src" / "stage_gate.py"
    authority_target.parent.mkdir(parents=True, exist_ok=True)
    authority_target.write_text("AUTHORITY = 'CANONICAL'\n", encoding="utf-8")

    law_target = canonical / "docs" / "law_a.md"
    law_target.parent.mkdir(parents=True, exist_ok=True)
    law_target.write_text("LAW-A\n", encoding="utf-8")

    for relative_path in ("src/stage_gate.py", "docs/law_a.md"):
        content = (canonical / relative_path).read_text(encoding="utf-8")
        live_target = root / relative_path
        live_target.parent.mkdir(parents=True, exist_ok=True)
        live_target.write_text(content, encoding="utf-8")
        baseline_target = baseline / relative_path
        baseline_target.parent.mkdir(parents=True, exist_ok=True)
        baseline_target.write_text(content, encoding="utf-8")

    canonical_root = build_canonical_root(canonical, authority_paths=["src/stage_gate.py"])
    canonical_sha = sha256_file(canonical / target)
    return root, canonical_root, canonical_sha



def decision_for(root: Path, path: str) -> tuple[object, TrustedEntry]:
    baseline = root / ".garvis" / "baseline" / path
    entry = TrustedEntry(
        path=path,
        sha256=sha256_file(baseline),
        baseline=path,
        auto_repair=True,
    )
    decision = next(
        item
        for item in build_plan(root, {entry.path: entry})
        if item.observation.path == path
    )
    return decision, entry


def verifier_for(canonical_root) -> object:
    def _verify(root: Path, target: str, candidate_sha: str) -> VerificationEvidence:
        del root
        return VerificationEvidence(
            oab_relationships_preserved=True,
            stage_gate_preserved=True,
            hyperq_verified=True,
            tests_pass=True,
            evidence_sha256=expected_evidence_sha256(
                target=target,
                candidate_sha=candidate_sha,
                root_hash=canonical_root.root_hash,
            ),
        )

    return _verify



def test_executor_refuses_nonsealed_disposition(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    target = root / "docs" / "law_a.md"
    target.write_text("DRIFT\n", encoding="utf-8")

    decision, entry = decision_for(root, "docs/law_a.md")

    with pytest.raises(RepairRefused, match="SEALED_REPAIR_REQUIRED"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=verifier_for(canonical_root),
        )

    assert target.read_text(encoding="utf-8") == "DRIFT\n"



def test_random_baseline_without_canonical_anchor_is_refused(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    random = root / "src" / "random.py"
    random.parent.mkdir(parents=True, exist_ok=True)
    random.write_text("CANONICAL-RANDOM\n", encoding="utf-8")

    baseline = root / ".garvis" / "baseline" / "src" / "random.py"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("CANONICAL-RANDOM\n", encoding="utf-8")

    entry = TrustedEntry(
        path="src/random.py",
        sha256=sha256_file(baseline),
        baseline="src/random.py",
        auto_repair=True,
    )
    random.write_text("DRIFT\n", encoding="utf-8")

    decision = force_sealed_decision(build_plan(root, {entry.path: entry})[0])

    with pytest.raises(RepairRefused, match="not independently anchored"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=verifier_for(canonical_root),
        )

    assert random.read_text(encoding="utf-8") == "DRIFT\n"



def test_authority_drift_can_restore_to_canonical(tmp_path: Path) -> None:
    root, canonical_root, canonical_sha = build_environment(
        tmp_path,
        target="src/stage_gate.py",
    )
    target = root / "src" / "stage_gate.py"
    target.write_text("AUTHORITY = 'DRIFTED'\n", encoding="utf-8")

    decision, entry = decision_for(root, "src/stage_gate.py")
    decision = force_sealed_decision(decision)

    result = sealed_auto_repair(
        root,
        decision,
        entry,
        canonical_root,
        expected_root_hash=canonical_root.root_hash,
        verifier=verifier_for(canonical_root),
    )

    assert result.repaired is True
    assert result.baseline_sha256 == canonical_sha
    assert result.canonical_sha256 == canonical_sha
    assert sha256_file(target) == canonical_sha
    assert result.candidate_sha256 == canonical_sha
    assert result.actual_sha256 == canonical_sha
    assert result.invariant_seal.seal_sha256
    assert result.invariant_seal.actual_sha256 == canonical_sha

    authority = compute_bundle(root, "authority", canonical_root.authority_paths)
    assert authority.sha256 == canonical_root.authority_bundle_sha256



def test_invalid_hyperq_evidence_hash_rolls_back(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    target = root / "docs" / "law_a.md"
    original = "PRESERVE-THIS\n"
    target.write_text(original, encoding="utf-8")

    decision, entry = decision_for(root, "docs/law_a.md")
    decision = force_sealed_decision(decision)

    def invalid_verifier(root: Path, target: str, candidate_sha: str) -> VerificationEvidence:
        del root, target, candidate_sha
        return VerificationEvidence(
            oab_relationships_preserved=True,
            stage_gate_preserved=True,
            hyperq_verified=True,
            tests_pass=True,
            evidence_sha256="invalid",
        )

    with pytest.raises(RepairRefused, match="evidence hash"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=invalid_verifier,
        )

    assert target.read_text(encoding="utf-8") == original



def test_executor_refuses_path_traversal_target(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("PRESERVE\n", encoding="utf-8")

    entry = TrustedEntry(
        path="../outside.md",
        sha256=sha256_file(root / ".garvis" / "baseline" / "docs" / "law_a.md"),
        baseline="docs/law_a.md",
        auto_repair=True,
    )
    decision = force_sealed_decision(build_plan(root, {entry.path: entry})[0])

    with pytest.raises(RepairRefused, match="path traversal"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=verifier_for(canonical_root),
        )

    assert outside.read_text(encoding="utf-8") == "PRESERVE\n"



def test_executor_refuses_symlink_target(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("PRESERVE\n", encoding="utf-8")
    target = root / "docs" / "law_a.md"
    target.unlink()
    os.symlink(outside, target)

    decision, entry = decision_for(root, "docs/law_a.md")
    decision = force_sealed_decision(decision)

    with pytest.raises(RepairRefused, match="symlink traversal"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=verifier_for(canonical_root),
        )

    assert outside.read_text(encoding="utf-8") == "PRESERVE\n"



def test_executor_refuses_path_identity_mismatch(tmp_path: Path) -> None:
    root, canonical_root, _ = build_environment(tmp_path)
    target = root / "docs" / "law_b.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("DRIFT\n", encoding="utf-8")

    entry = TrustedEntry(
        path="docs/law_b.md",
        sha256=sha256_file(root / ".garvis" / "baseline" / "docs" / "law_a.md"),
        baseline="docs/law_a.md",
        auto_repair=True,
    )
    decision = force_sealed_decision(build_plan(root, {entry.path: entry})[0])

    with pytest.raises(RepairRefused, match="path identity mismatch"):
        sealed_auto_repair(
            root,
            decision,
            entry,
            canonical_root,
            expected_root_hash=canonical_root.root_hash,
            verifier=verifier_for(canonical_root),
        )

    assert target.read_text(encoding="utf-8") == "DRIFT\n"
