# DIRECTIVE: GARVIS local eyes

Creator / conceptual architecture: Adrien D. Thomas / ProCityHub.

## Objective
Connect local image pixels to the existing VisualPathway and HypercubeBrainEngine.

## Current state
A callable LocalEyes bridge and command-line runner are implemented. The runner
owns a brain engine instance; it is not automatically loaded by the conversational
assistant or the persistent heartbeat service. No Flock account, feed, or camera
has been connected. This module makes no network requests.

## What This Means
Images become a 6 by 6 grid. Each cell records normalized brightness, color spread,
spatial variation, and mean absolute RGB change from the previous comparable frame.
Frames are sampled to 36 by 36 pixels before these measurements. Small changes can
be lost in downsampling; lighting or camera movement can also change pixels.
The existing visual pathway converts these measurements into observations consumed
by the brain engine. All 36 cells remain in the same `vision` evidence group.
Confidence 1.0 describes the computed measurements, not object identification.

A first frame, source change, dimension change, or synthetic/real mode change
starts a baseline: pixel_change is null. The legacy pathway's numeric motion
channel receives zero for baseline and pixel change for comparable frames;
consumers must check comparison_status before interpreting that channel.
Depth and semantic object identity remain undefined. Pixel differences do not
establish object motion, identity, intent, consciousness, or physical causation.
The existing OAB law is unchanged; its performance status remains
HYPOTHESIS_UNDER_TEST. This bridge adds no phi-based performance claim.

## Completed items
- Local image decoding, EXIF orientation, size limits, and canonical pixel hashes.
- Existing visual pathway and brain heartbeat integration.
- Bounded memory and JSONL output with UTC processing timestamps and provenance.
- Explicit, bounded desktop webcam option; device release on capture failure.
- Synthetic demo clearly marked SYNTHETIC_TEST.

## Run
From the repository root, install the optional image dependency:

```sh
python -m pip install -r requirements-local-eyes.txt
```

Run a two-frame synthetic demonstration (no camera is opened):

```sh
PYTHONPATH=src python -m hypercube_brain.local_eyes --demo
```

Supply ordered local images from the same camera/viewpoint:

```sh
PYTHONPATH=src python -m hypercube_brain.local_eyes --images before.jpg after.jpg
```

Desktop camera support requires `opencv-python>=4.10,<5` and a working local
camera. Explicitly authorize eight frames from device zero:

```sh
PYTHONPATH=src python -m hypercube_brain.local_eyes --camera 0 --allow-camera --frames 8
```

On Termux use image files exported by your camera app. Android direct camera
capture is not implemented by this module. URL streams and camera discovery are
not supported. No camera is opened at import or by the demo.

For a running Python application, retain one LocalEyes instance and call its
observe(image, source="your-stable-source-id") method for each decoded PIL image.
The object exposes its brain instance as `.brain`. Keep sources stable and never
mix different viewpoints under one source name. The image-sequence CLI assumes
all supplied files are from one viewpoint.

JSONL is emitted to stdout; there is no automatic upload or disk recording.
If you redirect output to a file, keep it private and outside the repository:
records contain image-derived data. observed_at is processing time, not capture
time. Local image origin and capture freshness are not authenticated.

## Pending items
Merge review, installation on the target device, real camera validation, and any
later persistent runtime integration. Flock integration requires a documented,
authorized data source and is not implemented here.

## Completion gate
26 local sensory/vision tests passed, including six new bridge tests. Controlled
PNG decoding and synthetic changes were tested. Camera error handling used a mock;
no physical camera or Flock service was tested. End-to-end deployment is complete
only after a real authorized frame reaches this bridge on the target device.

## Review
The bridge does not execute actions, recognize faces, read plates, identify people,
or contact remote services. Invalid files fail closed. A source/dimension reset
prevents comparing incompatible frames. Camera mode requires explicit activation
and a 1..144 frame bound; OS/device reads can still block depending on the driver.
No automated security review or hardware test should be inferred from these checks.

## Next command
After checking out this change and installing the image dependency:

```sh
PYTHONPATH=src python -m hypercube_brain.local_eyes --demo
```
