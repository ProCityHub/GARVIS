from __future__ import annotations

import pytest

from garvis.autonomous_repair_runner import (
    _extract_patch,
    _validate_patch,
)


def test_extract_patch_from_fence() -> None:
    response = """```diff
diff --git a/src/garvis/example.py b/src/garvis/example.py
--- a/src/garvis/example.py
+++ b/src/garvis/example.py
@@ -1 +1 @@
-old
+new
```"""
    patch = _extract_patch(response)
    assert patch is not None
    assert patch.startswith("diff --git ")


def test_no_patch_needed() -> None:
    assert _extract_patch("NO_PATCH_NEEDED") is None


def test_blocks_governance_path() -> None:
    patch = """diff --git a/src/garvis/thanos_mode.py b/src/garvis/thanos_mode.py
--- a/src/garvis/thanos_mode.py
+++ b/src/garvis/thanos_mode.py
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(RuntimeError):
        _validate_patch(patch)


def test_blocks_outside_scope() -> None:
    patch = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(RuntimeError):
        _validate_patch(patch)


def test_allows_adapter_source() -> None:
    patch = """diff --git a/src/garvis/provider_bridge.py b/src/garvis/provider_bridge.py
--- a/src/garvis/provider_bridge.py
+++ b/src/garvis/provider_bridge.py
@@ -1 +1 @@
-old
+new
"""
    assert _validate_patch(patch) == ["src/garvis/provider_bridge.py"]

def test_candidate_models_are_deduplicated(monkeypatch) -> None:
    from garvis.autonomous_repair_runner import _candidate_models

    monkeypatch.setenv(
        "GARVIS_REPAIR_MODELS",
        "model/a,model/b,model/a",
    )
    monkeypatch.setenv("GARVIS_MODEL", "model/b")

    assert _candidate_models("model/a")[:2] == [
        "model/a",
        "model/b",
    ]


def test_xai_becomes_candidate_when_configured(monkeypatch) -> None:
    from garvis.autonomous_repair_runner import _candidate_models

    for name in (
        "GARVIS_REPAIR_MODELS",
        "GARVIS_REPAIR_MODEL",
        "GARVIS_RESEARCH_MODEL",
        "GARVIS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("XAI_API_KEY", "configured-for-test")

    assert "compatible/grok-4.5" in _candidate_models()

def test_candidate_models_are_deduplicated(monkeypatch) -> None:
    from garvis.autonomous_repair_runner import _candidate_models

    monkeypatch.setenv(
        "GARVIS_REPAIR_MODELS",
        "model/a,model/b,model/a",
    )
    monkeypatch.setenv("GARVIS_MODEL", "model/b")

    assert _candidate_models("model/a")[:2] == [
        "model/a",
        "model/b",
    ]


def test_xai_becomes_candidate_when_configured(monkeypatch) -> None:
    from garvis.autonomous_repair_runner import _candidate_models

    for name in (
        "GARVIS_REPAIR_MODELS",
        "GARVIS_REPAIR_MODEL",
        "GARVIS_RESEARCH_MODEL",
        "GARVIS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("XAI_API_KEY", "configured-for-test")

    assert "compatible/grok-4.5" in _candidate_models()
