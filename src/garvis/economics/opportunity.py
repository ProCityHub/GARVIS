"""Opportunity contracts for jobs, services, products, and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OpportunityKind(str, Enum):
    JOB = "job"
    SERVICE = "service"
    DIGITAL_PRODUCT = "digital_product"
    CONTRACT = "contract"


class OpportunityStatus(str, Enum):
    DISCOVERED = "discovered"
    VERIFIED = "verified"
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    WON = "won"
    LOST = "lost"


def _unit(name: str, value: float) -> float:
    checked = float(value)
    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return checked


@dataclass(frozen=True)
class Opportunity:
    """One evidence-scored potential source of revenue."""

    opportunity_id: str
    title: str
    kind: OpportunityKind
    source_url: str
    gross_revenue: float
    estimated_cost: float
    estimated_hours: float
    value_created: float
    trust: float
    demand: float
    evidence: float
    risk: float
    status: OpportunityStatus = OpportunityStatus.DISCOVERED
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.opportunity_id.strip():
            raise ValueError("opportunity_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.source_url.strip():
            raise ValueError("source_url must not be empty")
        if self.gross_revenue < 0.0:
            raise ValueError("gross_revenue must not be negative")
        if self.estimated_cost < 0.0:
            raise ValueError("estimated_cost must not be negative")
        if self.estimated_hours <= 0.0:
            raise ValueError("estimated_hours must be greater than zero")
        _unit("value_created", self.value_created)
        _unit("trust", self.trust)
        _unit("demand", self.demand)
        _unit("evidence", self.evidence)
        _unit("risk", self.risk)

    @property
    def expected_profit(self) -> float:
        return self.gross_revenue - self.estimated_cost

    @property
    def expected_margin(self) -> float:
        if self.gross_revenue <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.expected_profit / self.gross_revenue))

    @property
    def expected_hourly_profit(self) -> float:
        return self.expected_profit / self.estimated_hours
