from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from garvis.stage_gate import sha256_payload

from .self_heal_projection import TrustedEntry
from .self_heal_root import CanonicalRoot, compute_bundle, sha256_file
from .self_heal_seal import SEALED_REPAIR_REQUIRED


class RepairRefused(RuntimeError):
    """Raised when a repair cannot be performed safely."""


@dataclass(frozen=True)
class VerificationEvidence:
    oab_relationships_preserved: bool
    stage_gate_preserved: bool
    hyperq_verified: bool
    tests_pass: bool
    evidence_sha256: str


@dataclass(frozen=True)
class RepairResult:
    path: str
    repaired: bool
    candidate_sha256: str
    evidence_sha256: str


Verifier = Callable[[Path, str, str], VerificationEvidence]



def expected_evidence_sha256(*, target: str, candidate_sha: str, root_hash: str) -> str:
    return sha256_payload(
        {
            "target": target,
            "candidate_sha": candidate_sha,
            "root_hash": root_hash,
        }
    )



def _repair_target(root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    return root / normalized



def _baseline_path(root: Path, baseline: str) -> Path:
    return root / ".garvis" / "baseline" / baseline



def _candidate_bytes(root: Path, entry: TrustedEntry, canonical_root: CanonicalRoot) -> bytes:
    baseline_path = _baseline_path(root, entry.baseline)
    canonical_path = canonical_root.root / entry.baseline

    if not baseline_path.is_file() or sha256_file(baseline_path) != entry.sha256:
        raise RepairRefused("baseline anchor hash mismatch")

    if not canonical_path.is_file() or sha256_file(canonical_path) != entry.sha256:
        raise RepairRefused("not independently anchored")

    return canonical_path.read_bytes()



def _restore(target: Path, original: bytes | None) -> None:
    if original is None:
        if target.exists():
            target.unlink()
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(original)



def sealed_auto_repair(
    root: Path,
    decision,
    entry: TrustedEntry,
    canonical_root: CanonicalRoot,
    *,
    expected_root_hash: str,
    verifier: Verifier,
) -> RepairResult:
    if getattr(decision, "disposition", "") != SEALED_REPAIR_REQUIRED:
        raise RepairRefused(SEALED_REPAIR_REQUIRED)

    if canonical_root.root_hash != expected_root_hash:
        raise RepairRefused("canonical root hash mismatch")

    candidate = _candidate_bytes(root, entry, canonical_root)
    candidate_sha = sha256_payload({"candidate": candidate.decode("utf-8")})
    target = _repair_target(root, entry.path)
    original = target.read_bytes() if target.exists() else None

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(candidate)

    try:
        evidence = verifier(root, entry.path, candidate_sha)
        if not (
            evidence.oab_relationships_preserved
            and evidence.stage_gate_preserved
            and evidence.hyperq_verified
            and evidence.tests_pass
        ):
            raise RepairRefused("verification failed")

        expected_evidence_hash = expected_evidence_sha256(
            target=entry.path,
            candidate_sha=candidate_sha,
            root_hash=canonical_root.root_hash,
        )
        if evidence.evidence_sha256 != expected_evidence_hash:
            raise RepairRefused("evidence hash mismatch")

        if entry.path in canonical_root.authority_paths:
            authority = compute_bundle(root, "authority", canonical_root.authority_paths)
            if authority.sha256 != canonical_root.authority_bundle_sha256:
                raise RepairRefused("authority bundle mismatch")

        return RepairResult(
            path=entry.path,
            repaired=True,
            candidate_sha256=candidate_sha,
            evidence_sha256=evidence.evidence_sha256,
        )
    except Exception:
        _restore(target, original)
        raise
