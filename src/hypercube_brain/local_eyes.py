"""Local pixels -> VisualPathway -> HypercubeBrainEngine.

Creator / conceptual architecture: Adrien D. Thomas / ProCityHub.
Pixel measurements are not semantic recognition or consciousness evidence.
No network sources are accepted. Camera capture requires an explicit CLI flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .core import HypercubeBrainEngine
from .visual_pathway import VisualFeature, VisualPathway


class LocalEyes:
    """Bounded, single-source image observation with no external action authority."""

    def __init__(self):
        self.pathway = VisualPathway()
        self.brain = HypercubeBrainEngine()
        self.previous = None
        self.previous_key = None
        self.previous_hash = None

    def observe(self, image, *, source: str, synthetic: bool = False):
        from PIL import Image, ImageChops, ImageStat

        if not source.strip():
            raise ValueError("source is required")
        if image.width < 6 or image.height < 6:
            raise ValueError("image must be at least 6 by 6 pixels")
        if image.width * image.height > 20_000_000:
            raise ValueError("image exceeds 20 megapixel limit")
        rgb = image.convert("RGB")
        digest = hashlib.sha256(
            f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
        ).hexdigest()
        sample = rgb.resize((36, 36), Image.Resampling.BOX)
        key = (source, rgb.size, synthetic)
        comparable = self.previous is not None and self.previous_key == key
        delta = ImageChops.difference(sample, self.previous) if comparable else None
        features = []
        cells = []
        for row in range(6):
            for col in range(6):
                box = (col * 6, row * 6, (col + 1) * 6, (row + 1) * 6)
                stats = ImageStat.Stat(sample.crop(box))
                intensity = sum(stats.mean) / (3 * 255)
                change = sum(ImageStat.Stat(delta.crop(box)).mean) / (3 * 255) if delta else None
                x, y = (col + 0.5) / 6, (row + 0.5) / 6
                label = f"pixel_cell_{row}_{col}; brightness={intensity:.6f}; semantics=UNDEFINED"
                features.append(VisualFeature(
                    feature_id=f"{digest}:{row}:{col}", label=label, x=x, y=y,
                    confidence=1.0, visual_field="left" if col < 2 else "right" if col > 3 else "central",
                    intensity=intensity, motion=change or 0.0,
                    detail=sum(stats.stddev) / (3 * 255),
                    color_signal=(max(stats.mean) - min(stats.mean)) / 255,
                    source=source,
                ))
                cells.append({"row": row, "column": col, "brightness": intensity, "pixel_change": change})
        percepts, collateral = self.pathway.process(features)
        observations = [self.pathway.to_brain_observation(p) for p in percepts]
        result = self.brain.heartbeat(
            claim="The supplied frame produced a 6 by 6 grid of pixel measurements.",
            observations=observations,
        )
        for history in (self.brain.episodes, self.brain.events):
            del history[:-89]
        record = {
            "schema_version": 1, "source": source,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "evidence_kind": "SYNTHETIC_TEST" if synthetic else "LOCAL_IMAGE_PIXELS",
            "frame_sha256": digest,
            "previous_frame_sha256": self.previous_hash if comparable else None,
            "width": rgb.width, "height": rgb.height,
            "comparison_status": "COMPARED" if comparable else "BASELINE",
            "cells": cells, "percepts": [asdict(p) for p in percepts],
            "collateral": [asdict(c) for c in collateral],
            "observations": [asdict(o) for o in observations],
            "brain_assessment": asdict(result), "external_action_allowed": False,
            "semantic_recognition": "UNDEFINED", "math_status": "HYPOTHESIS_UNDER_TEST",
            "motion_note": "motion channel carries pixel change, not verified object motion",
        }
        self.previous, self.previous_key, self.previous_hash = sample, key, digest
        return record


def read_image(path):
    from PIL import Image, ImageOps

    if not Path(path).is_file():
        raise ValueError("image source must be a local file")
    with Image.open(path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("image exceeds 20 megapixel limit")
        return ImageOps.exif_transpose(image).convert("RGB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--images", nargs="+", metavar="PATH")
    source.add_argument("--camera", type=int, metavar="LOCAL_INDEX")
    parser.add_argument("--allow-camera", action="store_true", help="explicitly authorize local camera activation")
    parser.add_argument("--frames", type=int, default=8, help="camera frames, 1..144")
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 144:
        parser.error("--frames must be between 1 and 144")
    if args.camera is not None and (args.camera < 0 or not args.allow_camera):
        parser.error("local --camera requires a nonnegative index and --allow-camera")
    eyes = LocalEyes()

    def emit(image, source, synthetic=False):
        print(json.dumps(eyes.observe(image, source=source, synthetic=synthetic)), flush=True)

    try:
        from PIL import Image

        if args.demo:
            first = Image.new("RGB", (60, 60), "black")
            second = first.copy()
            second.paste((255, 255, 255), (0, 0, 10, 10))
            for frame in (first, second):
                emit(frame, "synthetic_demo", True)
        elif args.images:
            for path in args.images:
                emit(read_image(path), "local_image_sequence")
        else:
            import cv2

            camera = cv2.VideoCapture(args.camera)
            try:
                if not camera.isOpened():
                    raise ValueError("local camera could not be opened")
                for _ in range(args.frames):
                    ok, bgr = camera.read()
                    if not ok:
                        raise ValueError("camera frame read failed")
                    emit(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), f"local_camera:{args.camera}")
            finally:
                camera.release()
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"Local eyes failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
