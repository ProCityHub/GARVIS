from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
class MathematicalInvariantSeal:
    path: str
    baseline_sha256: str
    canonical_sha256: str
    candidate_sha256: str
    actual_sha256: str
    expected_root_hash: str
    evidence_sha256: str
    seal_sha256: str


@dataclass(frozen=True)
class RepairResult:
    path: str
    repaired: bool
    baseline_sha256: str
    canonical_sha256: str
    candidate_sha256: str
    actual_sha256: str
    evidence_sha256: str
    invariant_seal: MathematicalInvariantSeal


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
    return _safe_path(root, relative_path)



def _baseline_path(root: Path, baseline: str) -> Path:
    return _safe_path(root / ".garvis" / "baseline", baseline)



def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()



def _safe_path(root: Path, relative_path: str) -> Path:
    if root.is_symlink():
        raise RepairRefused("symlink traversal refused")

    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute():
        raise RepairRefused("path traversal refused")

    parts: list[str] = []
    for part in pure.parts:
        if part in {"", "."}:
            raise RepairRefused("ambiguous path refused")
        if part == "..":
            raise RepairRefused("path traversal refused")
        parts.append(part)

    if not parts:
        raise RepairRefused("empty path refused")

    current = root
    for part in parts:
        if current.is_symlink():
            raise RepairRefused("symlink traversal refused")
        current = current / part
        if current.exists() and current.is_symlink():
            raise RepairRefused("symlink traversal refused")

    return current



def _candidate_material(
    root: Path,
    entry: TrustedEntry,
    canonical_root: CanonicalRoot,
) -> tuple[bytes, str, str]:
    baseline_path = _baseline_path(root, entry.baseline)
    canonical_path = _safe_path(canonical_root.root, entry.baseline)

    if not baseline_path.is_file():
        raise RepairRefused("baseline anchor missing")
    baseline_sha256 = sha256_file(baseline_path)
    if baseline_sha256 != entry.sha256:
        raise RepairRefused("baseline anchor hash mismatch")

    if not canonical_path.is_file():
        raise RepairRefused("canonical anchor missing")
    canonical_sha256 = sha256_file(canonical_path)
    if canonical_sha256 != entry.sha256:
        raise RepairRefused("not independently anchored")

    candidate = canonical_path.read_bytes()
    candidate_sha256 = _sha256_bytes(candidate)
    if candidate_sha256 != baseline_sha256 or candidate_sha256 != canonical_sha256:
        raise RepairRefused("mathematical invariant mismatch")

    return candidate, baseline_sha256, canonical_sha256



def _restore(target: Path, original: bytes | None) -> None:
    if original is None:
        if target.exists():
            target.unlink()
        return
    _atomic_write(target, original)



def _atomic_write(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise



def _seal_record(
    *,
    path: str,
    baseline_sha256: str,
    canonical_sha256: str,
    candidate_sha256: str,
    actual_sha256: str,
    expected_root_hash: str,
    evidence_sha256: str,
) -> MathematicalInvariantSeal:
    payload = {
        "actual_sha256": actual_sha256,
        "baseline_sha256": baseline_sha256,
        "candidate_sha256": candidate_sha256,
        "canonical_sha256": canonical_sha256,
        "evidence_sha256": evidence_sha256,
        "expected_root_hash": expected_root_hash,
        "path": path,
    }
    return MathematicalInvariantSeal(
        path=path,
        baseline_sha256=baseline_sha256,
        canonical_sha256=canonical_sha256,
        candidate_sha256=candidate_sha256,
        actual_sha256=actual_sha256,
        expected_root_hash=expected_root_hash,
        evidence_sha256=evidence_sha256,
        seal_sha256=sha256_payload(payload),
    )



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

    candidate, baseline_sha256, canonical_sha256 = _candidate_material(root, entry, canonical_root)
    candidate_sha = _sha256_bytes(candidate)
    target = _repair_target(root, entry.path)
    if target.exists() and not target.is_file():
        raise RepairRefused("target is not a regular file")
    original = target.read_bytes() if target.exists() else None

    _atomic_write(target, candidate)

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

        actual_sha256 = sha256_file(target)
        if (
            actual_sha256 != candidate_sha
            or actual_sha256 != baseline_sha256
            or actual_sha256 != canonical_sha256
        ):
            raise RepairRefused("mathematical invariant mismatch")

        invariant_seal = _seal_record(
            path=entry.path,
            baseline_sha256=baseline_sha256,
            canonical_sha256=canonical_sha256,
            candidate_sha256=candidate_sha,
            actual_sha256=actual_sha256,
            expected_root_hash=canonical_root.root_hash,
            evidence_sha256=evidence.evidence_sha256,
        )
        return RepairResult(
            path=entry.path,
            repaired=True,
            baseline_sha256=baseline_sha256,
            canonical_sha256=canonical_sha256,
            candidate_sha256=candidate_sha,
            actual_sha256=actual_sha256,
            evidence_sha256=evidence.evidence_sha256,
            invariant_seal=invariant_seal,
        )
    except Exception:
        _restore(target, original)
        raise
