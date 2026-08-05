"""Bounded, read-only evidence examiner for the GARVIS project.

Abaddon Scout V1 examines only an explicitly supplied evidence directory.
Importing this module performs no scan, file creation, network operation,
message transmission, Git operation, or protected-system action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "procityhub.garvis.abaddon-scout-report.v1"
REPORT_NAME = "REPORT.json"
MANIFEST_NAME = "manifest.sha256"
ARCHIVE_NAME = "ABADDON_SCOUT_V1.zip"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
BUFFER_SIZE = 1024 * 1024


class ScoutError(RuntimeError):
    """Base exception for a bounded Scout failure."""


class ContainmentError(ScoutError):
    """Raised when a path escapes or weakens the approved evidence boundary."""


class OutputCollisionError(ScoutError):
    """Raised when a Scout output would overwrite an existing file."""


class EvidenceChangedError(ScoutError):
    """Raised when evidence changes between examination and packaging."""


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _normalize_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ContainmentError(f"invalid relative evidence path: {value!r}")

    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ContainmentError(f"unsafe relative evidence path: {value!r}")

    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        raise ContainmentError(f"absolute evidence path is prohibited: {value!r}")

    return candidate.as_posix()


def _normalize_expected(
    expected_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}

    for raw_path, raw_digest in (expected_hashes or {}).items():
        relative = _normalize_relative_path(str(raw_path))
        digest = str(raw_digest).lower()

        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ScoutError(f"invalid SHA-256 digest for {relative!r}")

        if relative in normalized:
            raise ScoutError(f"duplicate expected path: {relative!r}")

        normalized[relative] = digest

    return normalized


def _resolve_evidence_root(evidence_root: str | Path) -> Path:
    supplied = Path(evidence_root).expanduser()

    if supplied.is_symlink():
        raise ContainmentError("the evidence root itself may not be a symbolic link")

    try:
        resolved = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ScoutError(f"evidence root does not exist: {supplied}") from exc

    if not resolved.is_dir():
        raise ScoutError(f"evidence root is not a directory: {resolved}")

    return resolved


def _iter_evidence_files(root: Path) -> list[Path]:
    discovered: list[Path] = []

    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)

        for directory_name in sorted(directory_names):
            directory = current_path / directory_name
            if directory.is_symlink():
                raise ContainmentError(
                    f"symbolic-link directory is prohibited: "
                    f"{directory.relative_to(root).as_posix()}"
                )

        directory_names[:] = sorted(directory_names)

        for file_name in sorted(file_names):
            file_path = current_path / file_name

            if file_path.is_symlink():
                raise ContainmentError(
                    f"symbolic-link file is prohibited: "
                    f"{file_path.relative_to(root).as_posix()}"
                )

            try:
                resolved = file_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ScoutError(f"evidence file disappeared: {file_path}") from exc

            if not _is_within(resolved, root):
                raise ContainmentError(f"evidence path escaped the root: {file_path}")

            if resolved.is_file():
                discovered.append(resolved)

    return sorted(discovered, key=lambda item: item.relative_to(root).as_posix())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def scan_evidence(
    evidence_root: str | Path,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Examine files contained by one explicitly supplied evidence root."""

    root = _resolve_evidence_root(evidence_root)
    expected = _normalize_expected(expected_hashes)
    entries: list[dict[str, object]] = []
    actual_paths: set[str] = set()
    contradictions: list[dict[str, str]] = []

    for file_path in _iter_evidence_files(root):
        relative = file_path.relative_to(root).as_posix()
        actual_digest = sha256_file(file_path)
        expected_digest = expected.get(relative)

        if expected_digest is None:
            status = "unverified"
        elif expected_digest == actual_digest:
            status = "verified"
        else:
            status = "mismatch"
            contradictions.append(
                {
                    "path": relative,
                    "expected_sha256": expected_digest,
                    "actual_sha256": actual_digest,
                }
            )

        actual_paths.add(relative)
        entries.append(
            {
                "path": relative,
                "size_bytes": file_path.stat().st_size,
                "sha256": actual_digest,
                "verification_status": status,
            }
        )

    missing_expected = sorted(set(expected) - actual_paths)
    verified_count = sum(
        item["verification_status"] == "verified" for item in entries
    )
    mismatch_count = sum(
        item["verification_status"] == "mismatch" for item in entries
    )
    unverified_count = sum(
        item["verification_status"] == "unverified" for item in entries
    )

    return {
        "schema": SCHEMA,
        "evidence_root_label": root.name,
        "files": entries,
        "expected_file_count": len(expected),
        "observed_file_count": len(entries),
        "verified_file_count": verified_count,
        "mismatch_file_count": mismatch_count,
        "unverified_file_count": unverified_count,
        "missing_expected_files": missing_expected,
        "contradictions": contradictions,
        "claim_boundaries": [
            "The Scout records files supplied inside the approved evidence root.",
            "A SHA-256 match verifies byte identity against the supplied digest.",
            "A hash alone does not prove authorship, ownership, infringement, "
            "platform conduct, scientific validity, or historical custody.",
            "Missing or conflicting evidence remains explicitly unresolved.",
            "The Scout has no access to account records or platform systems.",
        ],
    }


def _manifest_bytes(report: Mapping[str, object]) -> bytes:
    lines = [
        f"{entry['sha256']}  {entry['path']}"
        for entry in report["files"]  # type: ignore[index]
    ]
    text = "\n".join(lines)
    if text:
        text += "\n"
    return text.encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_verified_evidence_member(
    archive: zipfile.ZipFile,
    root: Path,
    entry: Mapping[str, object],
) -> None:
    relative = str(entry["path"])
    expected_digest = str(entry["sha256"])
    source = root / relative
    resolved = source.resolve(strict=True)

    if source.is_symlink() or not _is_within(resolved, root):
        raise ContainmentError(f"evidence path changed containment: {relative}")

    digest = hashlib.sha256()
    info = _zip_info(f"evidence/{relative}")

    with source.open("rb") as input_handle:
        with archive.open(info, "w") as output_handle:
            while chunk := input_handle.read(BUFFER_SIZE):
                digest.update(chunk)
                output_handle.write(chunk)

    if digest.hexdigest() != expected_digest:
        raise EvidenceChangedError(
            f"evidence changed during package generation: {relative}"
        )


def build_scout_package(
    evidence_root: str | Path,
    output_dir: str | Path,
    expected_hashes: Mapping[str, str] | None = None,
    *,
    generated_at: str | None = None,
    creator: str = "Adrien D. Thomas",
) -> dict[str, str]:
    """Generate REPORT.json, manifest.sha256, and a deterministic Scout ZIP."""

    root = _resolve_evidence_root(evidence_root)
    output_supplied = Path(output_dir).expanduser()

    if output_supplied.is_symlink():
        raise ContainmentError("the output directory may not be a symbolic link")

    output_candidate = output_supplied.resolve(strict=False)
    if _is_within(output_candidate, root):
        raise ContainmentError("the Scout output directory must be outside evidence")

    output_supplied.mkdir(parents=True, exist_ok=True)
    output = output_supplied.resolve(strict=True)

    if output.is_symlink() or not output.is_dir():
        raise ContainmentError("invalid Scout output directory")

    report_path = output / REPORT_NAME
    manifest_path = output / MANIFEST_NAME
    archive_path = output / ARCHIVE_NAME

    for target in (report_path, manifest_path, archive_path):
        if target.exists() or target.is_symlink():
            raise OutputCollisionError(f"Scout output already exists: {target}")

    report = scan_evidence(root, expected_hashes)
    report["creator"] = creator
    report["generated_at"] = generated_at or datetime.now(
        timezone.utc
    ).isoformat().replace("+00:00", "Z")
    report["package_contents"] = [
        REPORT_NAME,
        MANIFEST_NAME,
        ARCHIVE_NAME,
    ]

    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    manifest_bytes = _manifest_bytes(report)

    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for entry in report["files"]:  # type: ignore[index]
            _write_verified_evidence_member(archive, root, entry)

        archive.writestr(_zip_info(REPORT_NAME), report_bytes)
        archive.writestr(_zip_info(MANIFEST_NAME), manifest_bytes)

    report_path.write_bytes(report_bytes)
    manifest_path.write_bytes(manifest_bytes)

    return {
        "report": str(report_path),
        "manifest": str(manifest_path),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
    }


def _load_expected_json(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScoutError("expected-hash JSON must contain an object")

    return {str(key): str(value) for key, value in payload.items()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bounded Abaddon Scout V1 evidence package."
    )
    parser.add_argument("evidence_root")
    parser.add_argument("output_dir")
    parser.add_argument("--expected-json")
    parser.add_argument("--generated-at")
    parser.add_argument("--creator", default="Adrien D. Thomas")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = build_scout_package(
        arguments.evidence_root,
        arguments.output_dir,
        _load_expected_json(arguments.expected_json),
        generated_at=arguments.generated_at,
        creator=arguments.creator,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
