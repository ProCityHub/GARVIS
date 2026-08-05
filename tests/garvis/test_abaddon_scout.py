from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from garvis.abaddon_scout import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    ContainmentError,
    build_scout_package,
    scan_evidence,
    sha256_file,
)

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "garvis"
    / "abaddon_scout.py"
)
FIXED_TIME = "2026-08-05T17:00:00Z"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_import_has_no_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    module_name = "garvis_abaddon_scout_import_safety_test"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert set(tmp_path.iterdir()) == before


def test_scan_hashes_files_in_deterministic_path_order(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "z.txt").write_bytes(b"z")
    (root / "a.txt").write_bytes(b"a")

    report = scan_evidence(root, {"a.txt": digest(b"a")})

    assert [item["path"] for item in report["files"]] == ["a.txt", "z.txt"]
    assert report["files"][0]["verification_status"] == "verified"
    assert report["files"][1]["verification_status"] == "unverified"
    assert report["observed_file_count"] == 2


def test_scan_rejects_symbolic_link_escape(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "escape.txt"

    try:
        os.symlink(outside, link)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable on this filesystem")

    with pytest.raises(ContainmentError):
        scan_evidence(root)


def test_expected_hash_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()

    with pytest.raises(ContainmentError):
        scan_evidence(root, {"../outside.txt": "0" * 64})


def test_missing_and_mismatched_expected_evidence_is_preserved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "present.txt").write_bytes(b"actual")

    report = scan_evidence(
        root,
        {
            "present.txt": digest(b"different"),
            "missing.txt": digest(b"missing"),
        },
    )

    assert report["mismatch_file_count"] == 1
    assert report["missing_expected_files"] == ["missing.txt"]
    assert report["contradictions"] == [
        {
            "path": "present.txt",
            "expected_sha256": digest(b"different"),
            "actual_sha256": digest(b"actual"),
        }
    ]


def test_package_contains_report_manifest_and_evidence(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    output = tmp_path / "output"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "alpha.txt").write_bytes(b"alpha")
    (nested / "beta.bin").write_bytes(b"\x00\x01beta")

    result = build_scout_package(
        root,
        output,
        {
            "alpha.txt": digest(b"alpha"),
            "nested/beta.bin": digest(b"\x00\x01beta"),
        },
        generated_at=FIXED_TIME,
    )

    report = json.loads((output / REPORT_NAME).read_text(encoding="utf-8"))
    manifest = (output / MANIFEST_NAME).read_text(encoding="utf-8")

    assert report["creator"] == "Adrien D. Thomas"
    assert report["generated_at"] == FIXED_TIME
    assert report["verified_file_count"] == 2
    beta_digest = digest(b"\x00\x01beta")
    assert manifest == (
        f"{digest(b'alpha')}  alpha.txt\n"
        f"{beta_digest}  nested/beta.bin\n"
    )
    assert result["archive_sha256"] == sha256_file(output / ARCHIVE_NAME)

    with zipfile.ZipFile(output / ARCHIVE_NAME) as archive:
        assert archive.namelist() == [
            "evidence/alpha.txt",
            "evidence/nested/beta.bin",
            REPORT_NAME,
            MANIFEST_NAME,
        ]
        assert archive.read("evidence/alpha.txt") == b"alpha"
        assert archive.read("evidence/nested/beta.bin") == b"\x00\x01beta"
        assert archive.read(REPORT_NAME) == (output / REPORT_NAME).read_bytes()
        assert archive.read(MANIFEST_NAME) == (output / MANIFEST_NAME).read_bytes()


def test_package_archive_is_deterministic_with_fixed_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "proof.txt").write_bytes(b"proof")

    first = build_scout_package(
        root,
        tmp_path / "first",
        generated_at=FIXED_TIME,
    )
    second = build_scout_package(
        root,
        tmp_path / "second",
        generated_at=FIXED_TIME,
    )

    assert first["archive_sha256"] == second["archive_sha256"]
    assert Path(first["archive"]).read_bytes() == Path(second["archive"]).read_bytes()


def test_output_directory_inside_evidence_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()

    with pytest.raises(ContainmentError):
        build_scout_package(root, root / "scout-output", generated_at=FIXED_TIME)
