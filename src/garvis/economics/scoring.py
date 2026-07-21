"""Deterministic opportunity scoring for ProCityHub."""

from __future__ import annotations

from dataclasses import dataclass

from .opportunity import Opportunity


@dataclass(frozen=True)
class OpportunityScore:
    """Explainable score with the factors used to produce it."""

    score: float
    normalized_core: float
    expected_margin: float
    demand: float
    evidence: float
    risk: float

    @property
    def recommendation(self) -> str:
        if self.score >= 0.55:
            return "review_for_approval"
        if self.score >= 0.25:
            return "research_more"
        return "reject_or_deprioritize"


def score_opportunity(opportunity: Opportunity) -> OpportunityScore:
    """Score an opportunity between zero and one.

    The 1.0 value signal and 0.6 trust signal form a 1.6 normalized core.
    Demand, evidence, expected margin, and inverse risk then gate the result.
    """

    normalized_core = (
        (opportunity.value_created * 1.0) + (opportunity.trust * 0.6)
    ) / 1.6
    score = (
        normalized_core
        * opportunity.demand
        * opportunity.evidence
        * opportunity.expected_margin
        * (1.0 - opportunity.risk)
    )
    return OpportunityScore(
        score=max(0.0, min(1.0, score)),
        normalized_core=normalized_core,
        expected_margin=opportunity.expected_margin,
        demand=opportunity.demand,
        evidence=opportunity.evidence,
        risk=opportunity.risk,
    )
