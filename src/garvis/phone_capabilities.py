"""Read-only discovery of capabilities accessible to the Termux sandbox."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhoneCapability:
    capability_id: str
    available: bool
    access: str
    risk: str
    approval: str
    evidence: str


_TOOLS = {
    "python": ("local_python", "approved workspace", "medium"),
    "git": ("git_repository", "repository changes", "medium"),
    "gh": ("github_cli", "network and repository changes", "high"),
    "curl": ("public_http_client", "network", "medium"),
    "termux-battery-status": ("battery_status", "read-only device status", "low"),
    "termux-location": ("precise_location", "sensitive sensor data", "high"),
    "termux-camera-photo": ("camera", "sensitive sensor and file write", "high"),
    "termux-microphone-record": ("microphone", "sensitive sensor and file write", "high"),
}


def scan_phone_capabilities() -> dict[str, object]:
    capabilities = []
    for command, (capability_id, access, risk) in _TOOLS.items():
        path = shutil.which(command)
        capabilities.append(
            asdict(
                PhoneCapability(
                    capability_id,
                    path is not None,
                    access,
                    risk,
                    "required before use",
                    f"command={command}; path={path or 'not installed'}",
                )
            )
        )
    roots = []
    for candidate in (Path.home(), Path("/sdcard/Download"), Path.home() / "storage/downloads"):
        roots.append(
            {
                "path": str(candidate),
                "exists": candidate.exists(),
                "readable": os.access(candidate, os.R_OK) if candidate.exists() else False,
                "writable": os.access(candidate, os.W_OK) if candidate.exists() else False,
            }
        )
    return {
        "scope": "Termux sandbox and Android paths already granted to Termux",
        "warning": "No recursive file reading or Android permission bypass is performed.",
        "capabilities": capabilities,
        "storage_roots": roots,
    }


def render_phone_capabilities() -> str:
    return json.dumps(scan_phone_capabilities(), indent=2, sort_keys=True)
