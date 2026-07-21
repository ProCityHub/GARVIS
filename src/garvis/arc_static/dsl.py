"""Deterministic DSL of grid-transformation primitives (Track A, module 1).

The vocabulary a program-synthesis search will compose to solve static ARC
puzzles (input grid -> output grid). Every primitive is pure, total on
valid grids, deterministic, and never mutates its input.

A grid is a tuple of tuples of ints (colors 0-9 by ARC convention, though
any ints are accepted). `as_grid` normalizes lists/NumPy input.

Governance (enforced by tests):
- Pure offline stdlib computation; no model calls, no I/O.
- Primitives either return a valid grid or raise DslError; never None,
  never a partial/ragged grid.
- The registry is the single source of truth the search engine (module 2)
  will enumerate; every registered primitive is tested.

Authorship: Adrien D. Thomas / ProCityHub. Origin: GARVIS ARC-AGI Solver
Framework design (DIRECTIVE-014), implemented under audit.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Sequence, Tuple

Grid = Tuple[Tuple[int, ...], ...]


class DslError(ValueError):
    """Raised on invalid grids or invalid primitive parameters."""


def as_grid(value: Any) -> Grid:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DslError("grid must be a 2D sequence of integers")
    rows = list(value)
    if not rows:
        raise DslError("grid must not be empty")
    width = None
    out = []
    for r, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise DslError(f"row {r} is not a sequence")
        cells = list(row)
        if not cells:
            raise DslError(f"row {r} is empty")
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise DslError(f"ragged grid at row {r}")
        for c, v in enumerate(cells):
            if isinstance(v, bool) or not isinstance(v, int):
                raise DslError(f"non-integer value at ({r}, {c}): {v!r}")
        out.append(tuple(cells))
    return tuple(out)


def background_color(g: Grid) -> int:
    counts = Counter(v for row in g for v in row)
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def identity(g: Grid) -> Grid:
    return as_grid(g)


def rot90(g: Grid) -> Grid:
    g = as_grid(g)
    return tuple(zip(*g[::-1]))


def rot180(g: Grid) -> Grid:
    return rot90(rot90(g))


def rot270(g: Grid) -> Grid:
    return rot90(rot180(g))


def flip_h(g: Grid) -> Grid:
    g = as_grid(g)
    return tuple(tuple(reversed(row)) for row in g)


def flip_v(g: Grid) -> Grid:
    g = as_grid(g)
    return tuple(reversed(g))


def transpose(g: Grid) -> Grid:
    g = as_grid(g)
    return tuple(zip(*g))


def upscale(g: Grid, factor: int) -> Grid:
    g = as_grid(g)
    if not isinstance(factor, int) or factor < 1:
        raise DslError("factor must be a positive integer")
    out = []
    for row in g:
        wide = tuple(v for v in row for _ in range(factor))
        out.extend([wide] * factor)
    return tuple(out)


def downscale(g: Grid, factor: int) -> Grid:
    g = as_grid(g)
    if not isinstance(factor, int) or factor < 1:
        raise DslError("factor must be a positive integer")
    if len(g) % factor or len(g[0]) % factor:
        raise DslError("grid dimensions not divisible by factor")
    return tuple(
        tuple(g[r][c] for c in range(0, len(g[0]), factor))
        for r in range(0, len(g), factor)
    )


def tile(g: Grid, ny: int, nx: int) -> Grid:
    g = as_grid(g)
    if min(nx, ny) < 1:
        raise DslError("tile counts must be positive")
    return tuple(tuple(row * nx) for _ in range(ny) for row in g)


def hconcat(a: Grid, b: Grid) -> Grid:
    a, b = as_grid(a), as_grid(b)
    if len(a) != len(b):
        raise DslError("hconcat requires equal heights")
    return tuple(ra + rb for ra, rb in zip(a, b))


def vconcat(a: Grid, b: Grid) -> Grid:
    a, b = as_grid(a), as_grid(b)
    if len(a[0]) != len(b[0]):
        raise DslError("vconcat requires equal widths")
    return a + b


def replace_color(g: Grid, old: int, new: int) -> Grid:
    g = as_grid(g)
    return tuple(tuple(new if v == old else v for v in row) for row in g)


def swap_colors(g: Grid, a: int, b: int) -> Grid:
    g = as_grid(g)
    return tuple(
        tuple(b if v == a else a if v == b else v for v in row) for row in g
    )


def keep_color(g: Grid, color: int, fill: int = 0) -> Grid:
    g = as_grid(g)
    return tuple(tuple(v if v == color else fill for v in row) for row in g)


def remove_color(g: Grid, color: int, fill: int = 0) -> Grid:
    g = as_grid(g)
    return tuple(tuple(fill if v == color else v for v in row) for row in g)


def map_colors(g: Grid, mapping: Dict[int, int]) -> Grid:
    g = as_grid(g)
    return tuple(tuple(mapping.get(v, v) for v in row) for row in g)


def shift(g: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    g = as_grid(g)
    h, w = len(g), len(g[0])
    out = [[fill] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g[r][c]
    return tuple(tuple(row) for row in out)


def crop_to_content(g: Grid) -> Grid:
    g = as_grid(g)
    bg = background_color(g)
    cells = [
        (r, c)
        for r, row in enumerate(g)
        for c, v in enumerate(row)
        if v != bg
    ]
    if not cells:
        return g
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    return tuple(
        tuple(g[r][c] for c in range(min(cols), max(cols) + 1))
        for r in range(min(rows), max(rows) + 1)
    )


def pad(g: Grid, n: int, fill: int = 0) -> Grid:
    g = as_grid(g)
    if not isinstance(n, int) or n < 0:
        raise DslError("pad width must be non-negative")
    w = len(g[0]) + 2 * n
    top = tuple(tuple([fill] * w) for _ in range(n))
    body = tuple(tuple([fill] * n) + row + tuple([fill] * n) for row in g)
    return top + body + top


def border(g: Grid, color: int) -> Grid:
    g = as_grid(g)
    h, w = len(g), len(g[0])
    return tuple(
        tuple(
            color if r in (0, h - 1) or c in (0, w - 1) else g[r][c]
            for c in range(w)
        )
        for r in range(h)
    )


def gravity_down(g: Grid) -> Grid:
    g = as_grid(g)
    bg = background_color(g)
    h, w = len(g), len(g[0])
    cols = []
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c] != bg]
        cols.append([bg] * (h - len(col)) + col)
    return tuple(tuple(cols[c][r] for c in range(w)) for r in range(h))


UNARY_PRIMITIVES: Dict[str, Callable[[Grid], Grid]] = {
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "crop_to_content": crop_to_content,
    "gravity_down": gravity_down,
    "upscale2": lambda g: upscale(g, 2),
    "upscale3": lambda g: upscale(g, 3),
    "tile2x2": lambda g: tile(g, 2, 2),
    "tile1x2": lambda g: tile(g, 1, 2),
    "tile2x1": lambda g: tile(g, 2, 1),
    "mirror_right": lambda g: hconcat(g, flip_h(g)),
    "mirror_down": lambda g: vconcat(g, flip_v(g)),
    "pad1": lambda g: pad(g, 1),
}

COLOR_RANGE = tuple(range(10))
PARAM_FAMILIES = {
    "replace_color": (
        replace_color,
        tuple((o, n) for o in COLOR_RANGE for n in COLOR_RANGE if o != n),
    ),
    "keep_color": (keep_color, tuple((c,) for c in COLOR_RANGE)),
    "downscale": (downscale, ((2,), (3,))),
    "border": (border, tuple((c,) for c in COLOR_RANGE)),
}


def registry_size() -> int:
    return len(UNARY_PRIMITIVES) + sum(
        len(params) for _, params in PARAM_FAMILIES.values()
    )
