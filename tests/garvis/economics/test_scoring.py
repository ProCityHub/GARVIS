import pytest

from garvis.economics import Opportunity, OpportunityKind, score_opportunity


def test_high_quality_opportunity_scores_for_review() -> None:
    opportunity = Opportunity(
        opportunity_id="drywall-qc-001",
        title="Drywall deficiency inspection",
        kind=OpportunityKind.SERVICE,
        source_url="https://example.com/opportunity/1",
        gross_revenue=500.0,
        estimated_cost=100.0,
        estimated_hours=4.0,
        value_created=0.95,
        trust=0.95,
        demand=0.9,
        evidence=0.95,
        risk=0.05,
    )

    result = score_opportunity(opportunity)

    assert result.score == pytest.approx(0.61731, rel=1e-3)
    assert result.recommendation == "review_for_approval"
    assert opportunity.expected_profit == pytest.approx(400.0)
    assert opportunity.expected_hourly_profit == pytest.approx(100.0)


def test_risky_low_evidence_opportunity_is_deprioritized() -> None:
    opportunity = Opportunity(
        opportunity_id="online-001",
        title="Unverified online offer",
        kind=OpportunityKind.JOB,
        source_url="https://example.com/opportunity/2",
        gross_revenue=1000.0,
        estimated_cost=200.0,
        estimated_hours=20.0,
        value_created=0.5,
        trust=0.2,
        demand=0.5,
        evidence=0.2,
        risk=0.9,
    )

    assert score_opportunity(opportunity).recommendation == "reject_or_deprioritize"
