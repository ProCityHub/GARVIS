"""Educational knowledge pack for GARVIS."""

from __future__ import annotations

from .registry import list_systems


def anatomy_software_learning_pack() -> str:
    sections = [
        "Human anatomy studies body structure at gross and microscopic levels.",
        "The body is organized from cells to tissues, organs, and interacting organ systems.",
        "GARVIS uses the 11 organ systems only as a functional software analogy, not as a claim of biological equivalence.",
        "",
    ]

    for definition in list_systems():
        sections.extend(
            [
                f"{definition.system.value}:",
                f"  biological role: {definition.biological_role}",
                f"  software role: {definition.software_role}",
                "  responsibilities: " + "; ".join(definition.responsibilities),
                "",
            ]
        )

    sections.extend(
        [
            "Voice path:",
            "  nervous system plans language;",
            "  respiratory system manages timing and audio flow;",
            "  muscular system executes articulation or synthesis;",
            "  integumentary boundary exposes microphone and speaker interfaces;",
            "  feedback returns through the nervous system for correction.",
        ]
    )
    return "\n".join(sections)
