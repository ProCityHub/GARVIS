from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")



def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({_normalize_path(path) for path in paths}))



def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def compute_bundle(root: Path, name: str, paths: Iterable[str]) -> Bundle:
    normalized_paths = _normalize_paths(paths)
    members: list[tuple[str, str]] = []

    for relative_path in normalized_paths:
        file_path = root / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
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
    all_files = [
        str(path.relative_to(root)).replace("\\", "/")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    root_hash = compute_bundle(root, "root", all_files).sha256
    normalized_authority = _normalize_paths(authority_paths)
    authority_sha256 = compute_bundle(root, "authority", normalized_authority).sha256
    return CanonicalRoot(
        root=root,
        root_hash=root_hash,
        authority_paths=normalized_authority,
        authority_bundle_sha256=authority_sha256,
    )
