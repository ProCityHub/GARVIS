"""ARC-3 perception modules for GARVIS. Module 1: frame_parser."""
from .frame_parser import (
    ArcObject,
    BoundingBox,
    FrameValidationError,
    ParsedFrame,
    Point,
    parse_frame,
)

__all__ = [
    "ArcObject",
    "BoundingBox",
    "FrameValidationError",
    "ParsedFrame",
    "Point",
    "parse_frame",
]
