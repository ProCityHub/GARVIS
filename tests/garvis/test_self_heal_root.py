from pathlib import Path

from garvis.self_heal_root import build_canonical_root, compute_bundle, sha256_file



def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "law_a.md"
    target.parent.mkdir(parents=True)
    target.write_text("LAW\n", encoding="utf-8")

    assert sha256_file(target) == sha256_file(target)



def test_compute_bundle_tracks_expected_files(tmp_path: Path) -> None:
    first = tmp_path / "docs" / "law_a.md"
    second = tmp_path / "src" / "stage_gate.py"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("LAW\n", encoding="utf-8")
    second.write_text("AUTHORITY = 'CANONICAL'\n", encoding="utf-8")

    bundle = compute_bundle(tmp_path, "authority", ["src/stage_gate.py", "docs/law_a.md"])

    assert bundle.paths == ("docs/law_a.md", "src/stage_gate.py")
    assert len(bundle.members) == 2



def test_build_canonical_root_captures_authority_bundle(tmp_path: Path) -> None:
    target = tmp_path / "src" / "stage_gate.py"
    target.parent.mkdir(parents=True)
    target.write_text("AUTHORITY = 'CANONICAL'\n", encoding="utf-8")

    canonical_root = build_canonical_root(tmp_path, authority_paths=["src/stage_gate.py"])

    authority = compute_bundle(tmp_path, "authority", ["src/stage_gate.py"])
    assert canonical_root.authority_bundle_sha256 == authority.sha256
    assert canonical_root.root_hash
