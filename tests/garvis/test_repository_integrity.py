from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".cfg", ".css", ".html", ".ini", ".js", ".json", ".md",
    ".py", ".rst", ".sh", ".toml", ".ts", ".tsx", ".txt",
    ".yaml", ".yml",
}

BOUNDARIES = ("<<<<<<< ", ">>>>>>> ")


def test_repository_has_no_unresolved_merge_boundaries() -> None:
    failures: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for number, line in enumerate(lines, 1):
            if line.startswith(BOUNDARIES):
                failures.append(
                    f"{path.relative_to(ROOT)}:{number}: {line}"
                )

    assert not failures, (
        "Unresolved Git merge boundaries detected:\n"
        + "\n".join(failures)
    )
