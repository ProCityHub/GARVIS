"""Economic doctrine and safety boundaries for GARVIS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EconomicDoctrine:
    """Stable operating rules for ProCityHub revenue work."""

    authority: str = "Adrien D. Thomas"
    mission: str = "Create verified value for ProCityHub through lawful, measurable work."
    bank_access_enabled: bool = False
    autonomous_spending_enabled: bool = False
    external_actions_require_approval: bool = True


DOCTRINE = EconomicDoctrine()

CORE_RULES: tuple[str, ...] = (
    "Money follows verified value.",
    "Evidence comes before claims.",
    "No guaranteed-income promises.",
    "No bank credentials, PINs, security answers, or one-time codes are stored.",
    "No application, contract, purchase, payment, transfer, or account change without Adrien's approval.",
    "Every opportunity records source, cost, expected revenue, risk, and outcome.",
)
