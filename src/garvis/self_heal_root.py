from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from garvis.stage_gate import sha256_payload


@dataclass(frozen=True)
class Bundle:
    name: str
    paths: tuple[str, ...]
    sha256: str
    members: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CanonicalRoot:
    root: Path
    root_hash: str
    authority_paths: tuple[str, ...]
    authority_bundle_sha256: str


def _normalize_path(path: str) -> str:
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.is_absolute():
        raise ValueError("absolute paths are not allowed")

    normalized_parts: list[str] = []
    for part in pure.parts:
        if part in {"", "."}:
            raise ValueError("ambiguous paths are not allowed")
        if part == "..":
            raise ValueError("path traversal is not allowed")
        normalized_parts.append(part)

    if not normalized_parts:
        raise ValueError("path must not be empty")

    return "/".join(normalized_parts)



def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({_normalize_path(path) for path in paths}))



def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _safe_member_path(root: Path, relative_path: str) -> Path:
    if root.is_symlink():
        raise ValueError("symlink roots are not allowed")

    current = root
    for part in _normalize_path(relative_path).split("/"):
        if current.is_symlink():
            raise ValueError("symlink traversal is not allowed")
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("symlink traversal is not allowed")

    if not current.is_file():
        raise FileNotFoundError(current)

    return current



def compute_bundle(root: Path, name: str, paths: Iterable[str]) -> Bundle:
    normalized_paths = _normalize_paths(paths)
    members: list[tuple[str, str]] = []

    for relative_path in normalized_paths:
        file_path = _safe_member_path(root, relative_path)
        members.append((relative_path, sha256_file(file_path)))

    digest = sha256_payload(
        {
            "name": name,
            "members": [
                {"path": relative_path, "sha256": sha256}
                for relative_path, sha256 in members
            ],
        }
    )
    return Bundle(
        name=name,
        paths=normalized_paths,
        sha256=digest,
        members=tuple(members),
    )



def build_canonical_root(root: Path, *, authority_paths: Iterable[str]) -> CanonicalRoot:
    if root.is_symlink():
        raise ValueError("symlink roots are not allowed")

    all_files = [
        str(path.relative_to(root)).replace("\\", "/")
        for path in sorted(root.rglob("*"))
        if not path.is_symlink() and path.is_file()
    ]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("canonical root contains symlinks")
    root_hash = compute_bundle(root, "root", all_files).sha256
    normalized_authority = _normalize_paths(authority_paths)
    authority_sha256 = compute_bundle(root, "authority", normalized_authority).sha256
    return CanonicalRoot(
        root=root,
        root_hash=root_hash,
        authority_paths=normalized_authority,
        authority_bundle_sha256=authority_sha256,
    )
