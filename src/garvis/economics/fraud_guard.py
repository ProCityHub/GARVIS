"""Fraud and scam screening for economic opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FraudRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCK = "block"


@dataclass(frozen=True)
class FraudAssessment:
    risk: FraudRisk
    signals: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return self.risk is FraudRisk.BLOCK


_PATTERNS: tuple[tuple[str, str], ...] = (
    ("guaranteed income", "guaranteed_income"),
    ("guaranteed earnings", "guaranteed_income"),
    ("pay an application fee", "upfront_fee"),
    ("training fee", "upfront_fee"),
    ("buy gift cards", "gift_cards"),
    ("cash the cheque", "fake_cheque"),
    ("cash the check", "fake_cheque"),
    ("forward the money", "money_mule"),
    ("receive and transfer", "money_mule"),
    ("send your password", "credential_request"),
    ("send your pin", "credential_request"),
    ("one-time code", "credential_request"),
)


def assess_fraud(
    description: str,
    *,
    upfront_fee: bool = False,
    asks_for_credentials: bool = False,
    asks_to_move_money: bool = False,
    guaranteed_income: bool = False,
) -> FraudAssessment:
    """Return a conservative, explainable fraud assessment."""

    normalized = " ".join(description.casefold().split())
    signals = {signal for phrase, signal in _PATTERNS if phrase in normalized}
    if upfront_fee:
        signals.add("upfront_fee")
    if asks_for_credentials:
        signals.add("credential_request")
    if asks_to_move_money:
        signals.add("money_mule")
    if guaranteed_income:
        signals.add("guaranteed_income")

    blocking = {"credential_request", "money_mule", "fake_cheque", "gift_cards"}
    if signals & blocking:
        risk = FraudRisk.BLOCK
    elif len(signals) >= 2:
        risk = FraudRisk.HIGH
    elif len(signals) == 1:
        risk = FraudRisk.MEDIUM
    else:
        risk = FraudRisk.LOW

    return FraudAssessment(risk=risk, signals=tuple(sorted(signals)))
