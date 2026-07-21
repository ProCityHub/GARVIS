"""Deterministic ARC-3 per-game memory (DIRECTIVE-011, module 6 of 6).

Persists what was learned about one game — confirmed goal colors, demoted
hazard colors, wall colors, and evidence-backed action moves — to a JSON
file, and restores it exactly.

Governance (implementation facts, enforced by tests):
- Pure offline stdlib computation; local file I/O only, no network.
- Stores learned observations only: colors and action displacements.
  Never stores secrets, API keys, or free-form text.
- Deterministic serialization: identical memory yields identical bytes.
- Versioned schema; corrupt or wrong-version files are rejected loudly,
  never silently coerced.
- Atomic writes: a crash mid-save cannot corrupt an existing file.

Authorship: Adrien D. Thomas / ProCityHub. Spec: GARVIS DIRECTIVE-011.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

SCHEMA_VERSION = 1


class GameMemoryError(ValueError):
    """Raised on invalid memory data or unreadable files."""


def _check_color_set(values, name: str) -> FrozenSet[int]:
    out = set()
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise GameMemoryError(f"{name} entries must be integers")
        out.add(v)
    return frozenset(out)


@dataclass(frozen=True)
class GameMemory:
    game_id: str
    confirmed_goal_colors: FrozenSet[int] = frozenset()
    demoted_colors: FrozenSet[int] = frozenset()
    wall_colors: FrozenSet[int] = frozenset()
    action_moves: Tuple[Tuple[int, Tuple[int, int]], ...] = ()

    def __post_init__(self):
        if not isinstance(self.game_id, str) or not self.game_id:
            raise GameMemoryError("game_id must be a non-empty string")
        object.__setattr__(
            self, "confirmed_goal_colors",
            _check_color_set(self.confirmed_goal_colors, "confirmed_goal_colors"))
        object.__setattr__(
            self, "demoted_colors",
            _check_color_set(self.demoted_colors, "demoted_colors"))
        object.__setattr__(
            self, "wall_colors",
            _check_color_set(self.wall_colors, "wall_colors"))
        moves = []
        for item in self.action_moves:
            try:
                action_id, (dr, dc) = item
            except Exception as exc:
                raise GameMemoryError(f"bad action_moves entry: {item!r}") from exc
            for v in (action_id, dr, dc):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise GameMemoryError("action_moves must contain integers")
            moves.append((action_id, (dr, dc)))
        moves.sort()
        object.__setattr__(self, "action_moves", tuple(moves))

    def moves_dict(self) -> Dict[int, Tuple[int, int]]:
        return {a: d for a, d in self.action_moves}

    # ----- serialization -----

    def to_json(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "game_id": self.game_id,
            "confirmed_goal_colors": sorted(self.confirmed_goal_colors),
            "demoted_colors": sorted(self.demoted_colors),
            "wall_colors": sorted(self.wall_colors),
            "action_moves": [[a, [d[0], d[1]]] for a, d in self.action_moves],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "GameMemory":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GameMemoryError(f"corrupt memory file: {exc}") from exc
        if not isinstance(payload, dict):
            raise GameMemoryError("memory file must contain a JSON object")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise GameMemoryError(
                f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
        try:
            return cls(
                game_id=payload["game_id"],
                confirmed_goal_colors=payload["confirmed_goal_colors"],
                demoted_colors=payload["demoted_colors"],
                wall_colors=payload["wall_colors"],
                action_moves=tuple(
                    (a, (d[0], d[1])) for a, d in payload["action_moves"]),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise GameMemoryError(f"malformed memory payload: {exc}") from exc


def save_memory(memory: GameMemory, directory: str) -> Path:
    """Atomically write memory to <directory>/<game_id>.json."""
    if not isinstance(memory, GameMemory):
        raise GameMemoryError("save_memory requires a GameMemory")
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / f"{memory.game_id}.json"
    fd, tmp_name = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(memory.to_json())
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def load_memory(directory: str, game_id: str) -> GameMemory:
    path = Path(directory) / f"{game_id}.json"
    if not path.is_file():
        raise GameMemoryError(f"no memory file for game {game_id!r} in {directory}")
    memory = GameMemory.from_json(path.read_text(encoding="utf-8"))
    if memory.game_id != game_id:
        raise GameMemoryError(
            f"memory file {path.name} claims game_id {memory.game_id!r}")
    return memory


def has_memory(directory: str, game_id: str) -> bool:
    return (Path(directory) / f"{game_id}.json").is_file()
