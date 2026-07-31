"""Deterministic object-level primitives for static ARC (Track A, module 1b).

The low-level functions operate on ParsedFrame/ArcObject records produced by
GARVIS's tested frame parser. Composite wrappers expose ordinary Grid -> Grid
operations so the existing depth-2 static search engine can verify them.

Governance:
- Pure offline stdlib computation; no I/O, network, model calls, or hidden state.
- No semantic role assignment (wall, hazard, goal, player).
- Registered wrappers are total on valid grids and deterministic.
- Search still verifies every returned program against all training pairs.

Authorship: Adrien D. Thomas / ProCityHub.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from garvis.arc3.frame_parser import (
    ArcObject,
    BoundingBox,
    ParsedFrame,
    Point,
    parse_frame,
)

Grid = Tuple[Tuple[int, ...], ...]


class ObjectPrimitiveError(ValueError):
    """Raised when an object operation receives invalid parameters."""


def _as_grid(value: Any) -> Grid:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ObjectPrimitiveError("grid must be a 2D sequence of integers")
    rows = list(value)
    if not rows:
        raise ObjectPrimitiveError("grid must not be empty")
    width = None
    out = []
    for r, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ObjectPrimitiveError(f"row {r} is not a sequence")
        cells = list(row)
        if not cells:
            raise ObjectPrimitiveError(f"row {r} is empty")
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise ObjectPrimitiveError(f"ragged grid at row {r}")
        for c, item in enumerate(cells):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ObjectPrimitiveError(
                    f"non-integer value at ({r}, {c}): {item!r}"
                )
        out.append(tuple(cells))
    return tuple(out)


def extract_objects(
    grid: Any,
    *,
    background: Optional[int] = None,
    connectivity: int = 4,
) -> ParsedFrame:
    return parse_frame(
        _as_grid(grid), background=background, connectivity=connectivity
    )


def filter_objects_by(
    frame: ParsedFrame,
    *,
    colors: Optional[Iterable[int]] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    touching_boundary: Optional[bool] = None,
    shape_signature: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Tuple[ArcObject, ...]:
    allowed = None if colors is None else frozenset(colors)
    if min_pixels is not None and min_pixels < 1:
        raise ObjectPrimitiveError("min_pixels must be at least 1")
    if max_pixels is not None and max_pixels < 1:
        raise ObjectPrimitiveError("max_pixels must be at least 1")
    if (
        min_pixels is not None
        and max_pixels is not None
        and min_pixels > max_pixels
    ):
        raise ObjectPrimitiveError("min_pixels must not exceed max_pixels")

    out = []
    for obj in frame.objects:
        if allowed is not None and obj.color not in allowed:
            continue
        if min_pixels is not None and obj.pixel_count < min_pixels:
            continue
        if max_pixels is not None and obj.pixel_count > max_pixels:
            continue
        if (
            touching_boundary is not None
            and obj.touches_boundary is not touching_boundary
        ):
            continue
        if shape_signature is not None and obj.shape_signature != shape_signature:
            continue
        out.append(obj)
    return tuple(out)


def largest_object(frame: ParsedFrame) -> Optional[ArcObject]:
    return max(
        frame.objects,
        key=lambda obj: (
            obj.pixel_count,
            -obj.bounding_box.min_row,
            -obj.bounding_box.min_col,
            -obj.color,
        ),
        default=None,
    )


def smallest_object(frame: ParsedFrame) -> Optional[ArcObject]:
    return min(
        frame.objects,
        key=lambda obj: (
            obj.pixel_count,
            obj.bounding_box.min_row,
            obj.bounding_box.min_col,
            obj.color,
            obj.shape_signature,
        ),
        default=None,
    )


def _make_object(
    source: ArcObject,
    cells: Iterable[Tuple[int, int]],
    *,
    height: int,
    width: int,
) -> ArcObject:
    normalized = tuple(sorted(set(cells)))
    if not normalized:
        raise ObjectPrimitiveError("an object must contain at least one cell")
    if any(r < 0 or c < 0 or r >= height or c >= width for r, c in normalized):
        raise ObjectPrimitiveError("transformed object leaves the frame")
    rows = [r for r, _ in normalized]
    cols = [c for _, c in normalized]
    box = BoundingBox(min(rows), min(cols), max(rows), max(cols))
    signature = tuple(
        sorted((r - box.min_row, c - box.min_col) for r, c in normalized)
    )
    return ArcObject(
        object_id=source.object_id,
        color=source.color,
        cells=tuple(Point(r, c) for r, c in normalized),
        pixel_count=len(normalized),
        bounding_box=box,
        centroid=(sum(rows) / len(rows), sum(cols) / len(cols)),
        shape_signature=signature,
        touches_boundary=(
            box.min_row == 0
            or box.min_col == 0
            or box.max_row == height - 1
            or box.max_col == width - 1
        ),
    )


def translate_object(
    obj: ArcObject,
    *,
    dr: int,
    dc: int,
    height: int,
    width: int,
) -> ArcObject:
    if isinstance(dr, bool) or not isinstance(dr, int):
        raise ObjectPrimitiveError("dr must be an integer")
    if isinstance(dc, bool) or not isinstance(dc, int):
        raise ObjectPrimitiveError("dc must be an integer")
    return _make_object(
        obj,
        ((p.row + dr, p.col + dc) for p in obj.cells),
        height=height,
        width=width,
    )


def reflect_object_h(
    obj: ArcObject, *, height: int, width: int
) -> ArcObject:
    left, right = obj.bounding_box.min_col, obj.bounding_box.max_col
    return _make_object(
        obj,
        ((p.row, left + right - p.col) for p in obj.cells),
        height=height,
        width=width,
    )


def reflect_object_v(
    obj: ArcObject, *, height: int, width: int
) -> ArcObject:
    top, bottom = obj.bounding_box.min_row, obj.bounding_box.max_row
    return _make_object(
        obj,
        ((top + bottom - p.row, p.col) for p in obj.cells),
        height=height,
        width=width,
    )


def reconstruct_from_objects(
    frame: ParsedFrame,
    objects: Optional[Iterable[ArcObject]] = None,
    *,
    crop: bool = False,
) -> Grid:
    selected = tuple(frame.objects if objects is None else objects)
    if crop and not selected:
        raise ObjectPrimitiveError("cannot crop an empty object selection")

    if crop:
        min_row = min(obj.bounding_box.min_row for obj in selected)
        min_col = min(obj.bounding_box.min_col for obj in selected)
        max_row = max(obj.bounding_box.max_row for obj in selected)
        max_col = max(obj.bounding_box.max_col for obj in selected)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
    else:
        min_row = min_col = 0
        height, width = frame.height, frame.width

    canvas = [[frame.background_color] * width for _ in range(height)]
    for obj in selected:
        for point in obj.cells:
            row, col = point.row - min_row, point.col - min_col
            if not (0 <= row < height and 0 <= col < width):
                raise ObjectPrimitiveError("object cell lies outside render bounds")
            canvas[row][col] = obj.color
    return tuple(tuple(row) for row in canvas)


def _scene(grid: Any) -> tuple[Grid, ParsedFrame]:
    normalized = _as_grid(grid)
    return normalized, extract_objects(normalized)


def keep_largest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    return original if obj is None else reconstruct_from_objects(frame, (obj,))


def keep_smallest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = smallest_object(frame)
    return original if obj is None else reconstruct_from_objects(frame, (obj,))


def crop_largest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    return original if obj is None else reconstruct_from_objects(frame, (obj,), crop=True)


def crop_smallest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = smallest_object(frame)
    return original if obj is None else reconstruct_from_objects(frame, (obj,), crop=True)


def remove_largest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    if obj is None:
        return original
    return reconstruct_from_objects(
        frame, tuple(x for x in frame.objects if x.object_id != obj.object_id)
    )


def remove_smallest_object(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = smallest_object(frame)
    if obj is None:
        return original
    return reconstruct_from_objects(
        frame, tuple(x for x in frame.objects if x.object_id != obj.object_id)
    )


def _replace_object(
    frame: ParsedFrame, original: ArcObject, replacement: ArcObject
) -> Grid:
    remaining = tuple(
        x for x in frame.objects if x.object_id != original.object_id
    )
    return reconstruct_from_objects(frame, remaining + (replacement,))


def move_largest_to_top_left(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    if obj is None:
        return original
    moved = translate_object(
        obj,
        dr=-obj.bounding_box.min_row,
        dc=-obj.bounding_box.min_col,
        height=frame.height,
        width=frame.width,
    )
    return _replace_object(frame, obj, moved)


def move_smallest_to_top_left(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = smallest_object(frame)
    if obj is None:
        return original
    moved = translate_object(
        obj,
        dr=-obj.bounding_box.min_row,
        dc=-obj.bounding_box.min_col,
        height=frame.height,
        width=frame.width,
    )
    return _replace_object(frame, obj, moved)


def reflect_largest_h(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    if obj is None:
        return original
    return _replace_object(
        frame,
        obj,
        reflect_object_h(obj, height=frame.height, width=frame.width),
    )


def reflect_largest_v(grid: Any) -> Grid:
    original, frame = _scene(grid)
    obj = largest_object(frame)
    if obj is None:
        return original
    return _replace_object(
        frame,
        obj,
        reflect_object_v(obj, height=frame.height, width=frame.width),
    )


OBJECT_UNARY_PRIMITIVES: dict[str, Callable[[Grid], Grid]] = {
    "object_crop_largest": crop_largest_object,
    "object_crop_smallest": crop_smallest_object,
    "object_keep_largest": keep_largest_object,
    "object_keep_smallest": keep_smallest_object,
    "object_move_largest_top_left": move_largest_to_top_left,
    "object_move_smallest_top_left": move_smallest_to_top_left,
    "object_reflect_largest_h": reflect_largest_h,
    "object_reflect_largest_v": reflect_largest_v,
    "object_remove_largest": remove_largest_object,
    "object_remove_smallest": remove_smallest_object,
}
