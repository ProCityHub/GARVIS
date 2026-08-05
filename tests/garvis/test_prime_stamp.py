from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "src" / "garvis" / "prime_stamp.py"
DANGEROUS_CALLS = {
    "accept",
    "bind",
    "connect",
    "input",
    "listen",
    "mkdir",
    "open",
    "Popen",
    "recv",
    "recvfrom",
    "remove",
    "run",
    "send",
    "sendall",
    "sendto",
    "socket",
    "start",
    "system",
    "Thread",
    "unlink",
    "write",
    "write_text",
}


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _unguarded_top_level_effects(tree: ast.Module) -> list[tuple[int, str]]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    effects: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_name = _dotted_name(node.func)
        if call_name.rsplit(".", 1)[-1] not in DANGEROUS_CALLS:
            continue

        current: ast.AST = node
        guarded_by_main = False
        inside_callable = False

        while current in parent:
            current = parent[current]
            if isinstance(
                current,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Lambda,
                    ast.ClassDef,
                ),
            ):
                inside_callable = True
                break
            if isinstance(current, ast.If):
                condition = ast.unparse(current.test)
                if "__name__" in condition and "__main__" in condition:
                    guarded_by_main = True

        if not inside_callable and not guarded_by_main:
            effects.append((node.lineno, call_name))

    return effects


def test_prime_stamp_uses_timezone_aware_utc() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "datetime.utcnow()" not in source
    assert "timezone.utc" in source


def test_prime_stamp_preserves_creator_attribution() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "Adrien D. Thomas" in source


def test_prime_stamp_has_no_unguarded_top_level_side_effects() -> None:
    assert _unguarded_top_level_effects(_tree()) == []


def test_prime_stamp_import_is_quiet_and_creates_no_files(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())

    module_name = "_garvis_prime_stamp_prototype_test"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert set(tmp_path.iterdir()) == before
