from garvis.economics import FraudRisk, assess_fraud


def test_credentials_or_money_movement_blocks_opportunity() -> None:
    assessment = assess_fraud(
        "Receive and transfer funds, then send your one-time code.",
    )

    assert assessment.risk is FraudRisk.BLOCK
    assert assessment.blocked is True
    assert "credential_request" in assessment.signals
    assert "money_mule" in assessment.signals


def test_clean_description_is_low_risk() -> None:
    assessment = assess_fraud("Paid drywall quality-control inspection for a verified builder.")

    assert assessment.risk is FraudRisk.LOW
    assert assessment.signals == ()
