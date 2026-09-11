"""Phase 19: loads the COMPLETE prediction pipeline (features + model + threshold +
policy + version metadata) as one object, so the API never has to reason about
which preprocessing matches which model version.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: F401
from src.fraudshield.models.risk_engine import RiskPolicy

REQUIRED_RAW_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


class FraudInferencePipeline:
    def __init__(self, model_dir: str = "data/features/models"):
        model_dir = Path(model_dir)
        config = json.load(open(model_dir / "model_config.json"))
        self.model_version = config["model_version"]
        self.threshold = config["decision_threshold"]
        self.feature_pipeline = joblib.load(Path("data/features") / "feature_pipeline.joblib")
        self.model = joblib.load(model_dir / config["model_file"])
        self.policy = RiskPolicy.from_model_config(config)
        self.loaded_at = time.time()

    def predict_one(self, transaction: dict) -> dict:
        t0 = time.perf_counter()
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in transaction]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        row = pd.DataFrame([transaction])
        X = self.feature_pipeline.transform(row).values
        prob = float(self.model.predict_proba(X)[0, 1])
        risk_level, reason = self.policy.decide(prob, amount=transaction.get("Amount"))
        decision = "BLOCK" if risk_level.name == "HIGH" else (
            "REVIEW" if risk_level.name == "MEDIUM" else "APPROVE"
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "fraud_probability": round(prob, 6),
            "risk_level": risk_level.value,
            "decision": decision,
            "decision_reason": reason,
            "model_version": self.model_version,
            "threshold_version": round(self.threshold, 6),
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        return [self.predict_one(tx) for tx in transactions]
