"""Law Zero loader.

The stamp is read before every other cycle step. Adrien D. Thomas is the
sole final authority. The body proposes only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CREATOR = "Adrien D. Thomas"

LAW_ZERO_TEXT = """\
Law 0: Governance Stamp (read before step 1, binds every step)

Creator: Adrien D. Thomas (Cree, Treaty 6 Territory), creator and owner of ProCityHub, the Lattice, GENESIS X and this journal.
Final authority: Adrien D. Thomas alone. No bot, routine or group can override, amend or remove this law.

As a lattice law: the creator is the origin point every lattice state is measured from. A cycle that breaks this law is not INTEGRATED, whatever its score.
As a governance law: merges, deletions, closures, sends and changes to this protocol happen only on the creator's word. When in doubt, a bot stops and hands off to him.
"""

_REQUIRED_MARKERS = (
    "Creator: Adrien D. Thomas",
    "Final authority: Adrien D. Thomas alone.",
    "No bot, routine or group can override, amend or remove this law.",
    "A cycle that breaks this law is not INTEGRATED, whatever its score.",
    "merges, deletions, closures, sends and changes to this protocol happen only on the creator's word.",
    "When in doubt, a bot stops and hands off to him.",
)


@dataclass(frozen=True)
class LawZeroCheck:
    """Result of reading the stamp. ``ok`` is false when the text is not the stamp."""

    ok: bool
    creator: str
    final_authority: str
    body_proposes_only: bool
    missing: tuple[str, ...]
    text: str
    source: str


def load_law_zero(path: str | Path | None = None) -> str:
    """Return the stamp text.

    A missing explicit path is returned as an empty string so the cycle
    stops. It is not replaced with a quieter success.
    """

    if path is None:
        return LAW_ZERO_TEXT
    stamp_path = Path(path)
    if not stamp_path.is_file():
        return ""
    return stamp_path.read_text(encoding="utf-8", errors="replace")


def validate_law_zero(text: str) -> LawZeroCheck:
    """Check that the creator remains the sole final authority."""

    missing = tuple(marker for marker in _REQUIRED_MARKERS if marker not in text)
    ok = not missing and CREATOR in text
    return LawZeroCheck(
        ok=ok,
        creator=CREATOR if ok else "",
        final_authority=CREATOR if ok else "",
        body_proposes_only=True,
        missing=missing,
        text=text,
        source="embedded" if text == LAW_ZERO_TEXT else "supplied",
    )
