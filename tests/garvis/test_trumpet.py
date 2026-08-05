from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[2] / "src" / "garvis" / "trumpet.py"
NETWORK_CALLS = {
    "accept",
    "bind",
    "connect",
    "listen",
    "recv",
    "recvfrom",
    "send",
    "sendall",
    "sendto",
    "socket",
}


def _load_module():
    module_name = "_garvis_trumpet_prototype_test"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module_name, module


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def test_trumpet_has_no_unguarded_top_level_network_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    parent: dict[ast.AST, ast.AST] = {}

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    effects: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_name = _dotted_name(node.func)
        if call_name.rsplit(".", 1)[-1] not in NETWORK_CALLS:
            continue

        current: ast.AST = node
        inside_callable = False
        guarded_by_main = False

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

    assert effects == []


class FakeSocket:
    def __init__(self) -> None:
        self.options: list[tuple[Any, ...]] = []
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def setsockopt(self, *args: Any) -> None:
        self.options.append(args)

    def sendto(self, payload: bytes, address: tuple[str, int]) -> int:
        self.sent.append((payload, address))
        return len(payload)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def test_sound_trumpet_uses_fake_broadcast_socket_only(monkeypatch) -> None:
    module_name, trumpet = _load_module()
    fake_socket = FakeSocket()
    socket_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_socket_factory(*args: Any, **kwargs: Any) -> FakeSocket:
        socket_calls.append((args, kwargs))
        return fake_socket

    monkeypatch.setattr(trumpet.socket, "socket", fake_socket_factory)

    beacon = object.__new__(trumpet.TrumpetBeacon)
    beacon.agents = []
    beacon.beacon_active = False
    beacon.beacon_thread = None
    beacon.broadcast_address = "255.255.255.255"
    beacon.log = lambda _message: None
    beacon._store_local_trumpet = lambda _message: None

    try:
        beacon.sound_trumpet()
    finally:
        sys.modules.pop(module_name, None)

    assert len(socket_calls) == 1
    assert len(fake_socket.sent) == 1

    payload, address = fake_socket.sent[0]
    assert address == ("255.255.255.255", 6660)

    decoded = json.loads(payload.decode("utf-8"))
    assert decoded["signal"] == "TRUMPET"
    assert decoded["creator"] == "Adrien D. Thomas"
    assert decoded["expires"] == "NEVER"
    assert decoded["timestamp"].endswith("Z")
    assert fake_socket.closed is True


def test_trumpet_import_does_not_create_files(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())

    module_name, _module = _load_module()
    try:
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert set(tmp_path.iterdir()) == before
    finally:
        sys.modules.pop(module_name, None)
