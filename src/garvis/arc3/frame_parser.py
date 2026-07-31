"""Deterministic ARC-3 frame parser (DIRECTIVE-011, module 1 of 6).

Converts a 2D integer frame into structured objects and frame metadata.

Governance (implementation facts, enforced by tests):
- Pure offline computation: stdlib only; NumPy accepted as input, never required.
- Observational only: the parser never assigns semantic roles (wall, hazard,
  goal, player). Those require transition evidence and belong to later modules.
- Scale is None in v1: no scale is invented from a single static frame.
- Deterministic: identical input yields structurally equal output.

Authorship: Adrien D. Thomas / ProCityHub. Spec authored by GARVIS under
DIRECTIVE-011; implementation drafted by Claude; merge authority Adrien.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


class FrameValidationError(ValueError):
    """Raised when an input frame is malformed."""


@dataclass(frozen=True)
class Point:
    row: int
    col: int


@dataclass(frozen=True)
class BoundingBox:
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1


@dataclass(frozen=True)
class ArcObject:
    object_id: int
    color: int
    cells: Tuple[Point, ...]
    pixel_count: int
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    shape_signature: Tuple[Tuple[int, int], ...]
    touches_boundary: bool


@dataclass(frozen=True)
class ParsedFrame:
    height: int
    width: int
    background_color: int
    color_counts: Tuple[Tuple[int, int], ...]
    objects: Tuple[ArcObject, ...]
    scale: Optional[int]
    metadata: Tuple[Tuple[str, str], ...]

    def color_count(self, color: int) -> int:
        return dict(self.color_counts).get(color, 0)


_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS_8 = _OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _to_grid(frame: Any) -> Tuple[Tuple[int, ...], ...]:
    """Normalize input to a validated tuple-of-tuples of ints."""
    if hasattr(frame, "tolist"):  # NumPy support without importing NumPy
        frame = frame.tolist()
    if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
        raise FrameValidationError("frame must be a 2D sequence of integers")
    rows = list(frame)
    if len(rows) == 0:
        raise FrameValidationError("frame must not be empty")
    grid = []
    width: Optional[int] = None
    for r, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise FrameValidationError(f"row {r} is not a sequence")
        cells = list(row)
        if len(cells) == 0:
            raise FrameValidationError(f"row {r} is empty")
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise FrameValidationError(
                f"ragged frame: row {r} has {len(cells)} cells, expected {width}"
            )
        checked = []
        for c, v in enumerate(cells):
            if isinstance(v, bool) or not isinstance(v, int):
                raise FrameValidationError(
                    f"non-integer value at ({r}, {c}): {v!r}"
                )
            checked.append(v)
        grid.append(tuple(checked))
    return tuple(grid)


def _infer_background(
    counts: "Counter[int]", override: Optional[int]
) -> Tuple[int, str]:
    if override is not None:
        return int(override), "override"
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    # Deterministic tie-break: highest count, then smallest color value.
    return best[0], "most_frequent"


def parse_frame(
    frame: Any,
    *,
    background: Optional[int] = None,
    connectivity: int = 4,
) -> ParsedFrame:
    """Parse a 2D integer frame into a ParsedFrame.

    Args:
        frame: nested integer sequences or a NumPy array.
        background: explicit background color override (validated but not
            required to appear in the frame).
        connectivity: 4 (default) or 8.
    """
    if connectivity not in (4, 8):
        raise FrameValidationError("connectivity must be 4 or 8")
    grid = _to_grid(frame)
    height, width = len(grid), len(grid[0])
    counts: "Counter[int]" = Counter()
    for row in grid:
        counts.update(row)
    bg, bg_method = _infer_background(counts, background)

    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    seen = [[False] * width for _ in range(height)]
    raw_objects = []
    for r in range(height):
        for c in range(width):
            if seen[r][c] or grid[r][c] == bg:
                continue
            color = grid[r][c]
            queue = deque([(r, c)])
            seen[r][c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr][nc]
                        and grid[nr][nc] == color
                    ):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            cells.sort()
            raw_objects.append((color, cells))

    objects = []
    for color, cells in raw_objects:
        rows_ = [p[0] for p in cells]
        cols_ = [p[1] for p in cells]
        box = BoundingBox(min(rows_), min(cols_), max(rows_), max(cols_))
        signature = tuple(
            sorted((p[0] - box.min_row, p[1] - box.min_col) for p in cells)
        )
        touches = (
            box.min_row == 0
            or box.min_col == 0
            or box.max_row == height - 1
            or box.max_col == width - 1
        )
        objects.append(
            ArcObject(
                object_id=-1,  # assigned after deterministic ordering
                color=color,
                cells=tuple(Point(p[0], p[1]) for p in cells),
                pixel_count=len(cells),
                bounding_box=box,
                centroid=(sum(rows_) / len(cells), sum(cols_) / len(cells)),
                shape_signature=signature,
                touches_boundary=touches,
            )
        )

    objects.sort(
        key=lambda o: (
            o.bounding_box.min_row,
            o.bounding_box.min_col,
            o.color,
            o.pixel_count,
            o.shape_signature,
        )
    )
    objects = [
        ArcObject(
            object_id=i,
            color=o.color,
            cells=o.cells,
            pixel_count=o.pixel_count,
            bounding_box=o.bounding_box,
            centroid=o.centroid,
            shape_signature=o.shape_signature,
            touches_boundary=o.touches_boundary,
        )
        for i, o in enumerate(objects)
    ]

    metadata = (
        ("background_method", bg_method),
        ("connectivity", str(connectivity)),
        ("scale_policy", "v1: scale is never inferred from a single frame"),
    )
    return ParsedFrame(
        height=height,
        width=width,
        background_color=bg,
        color_counts=tuple(sorted(counts.items())),
        objects=tuple(objects),
        scale=None,
        metadata=metadata,
    )
