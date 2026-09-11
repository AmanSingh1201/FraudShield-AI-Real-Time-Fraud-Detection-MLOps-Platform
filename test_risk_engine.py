from src.fraudshield.models.risk_engine import RiskPolicy, RiskLevel


def make_policy():
    return RiskPolicy(review_threshold=0.02, block_threshold=0.05)


def test_low_risk_below_review_threshold():
    policy = make_policy()
    level, _ = policy.decide(0.001)
    assert level == RiskLevel.LOW


def test_medium_risk_between_thresholds():
    policy = make_policy()
    level, _ = policy.decide(0.03)
    assert level == RiskLevel.MEDIUM


def test_high_risk_at_or_above_block_threshold():
    policy = make_policy()
    level, _ = policy.decide(0.05)
    assert level == RiskLevel.HIGH
    level, _ = policy.decide(0.9)
    assert level == RiskLevel.HIGH


def test_boundary_exactly_at_review_threshold_is_medium():
    policy = make_policy()
    level, _ = policy.decide(0.02)
    assert level == RiskLevel.MEDIUM


def test_high_amount_business_rule_can_escalate_low_probability():
    policy = make_policy()
    # probability below review, but not below review*0.5, with large amount
    level, reason = policy.decide(0.011, amount=10000)
    assert level == RiskLevel.MEDIUM
    assert "business rule" in reason


def test_from_model_config():
    cfg = {"decision_threshold": 0.05, "review_threshold": 0.02}
    policy = RiskPolicy.from_model_config(cfg)
    assert policy.block_threshold == 0.05
    assert policy.review_threshold == 0.02
