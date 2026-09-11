"""Phase 16: risk decision engine.

Deliberately separate from the ML model: the model outputs a probability, this
layer maps probability -> business decision. Changing review-queue capacity or
risk appetite should never require retraining a model — just editing this
config-driven layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW_RISK"
    MEDIUM = "MEDIUM_RISK_REVIEW"
    HIGH = "HIGH_RISK_BLOCK"


@dataclass
class RiskPolicy:
    """Two thresholds define three bands. Loaded from model_config.json so the
    policy can change without a code deploy."""
    review_threshold: float   # probability >= this -> at least REVIEW
    block_threshold: float    # probability >= this -> BLOCK
    max_amount_auto_approve: float = 5000.0  # simple business rule example

    def decide(self, fraud_probability: float, amount: float | None = None) -> tuple[RiskLevel, str]:
        if fraud_probability >= self.block_threshold:
            return RiskLevel.HIGH, "probability >= block_threshold"
        if fraud_probability >= self.review_threshold:
            return RiskLevel.MEDIUM, "probability >= review_threshold"
        if amount is not None and amount > self.max_amount_auto_approve and fraud_probability >= self.review_threshold * 0.5:
            return RiskLevel.MEDIUM, "high amount + elevated probability business rule"
        return RiskLevel.LOW, "probability below review_threshold"

    @classmethod
    def from_model_config(cls, config: dict) -> "RiskPolicy":
        return cls(
            review_threshold=config["review_threshold"],
            block_threshold=config["decision_threshold"],
        )
