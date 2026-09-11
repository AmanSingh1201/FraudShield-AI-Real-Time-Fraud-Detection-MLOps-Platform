"""Feature engineering, implemented as a single sklearn-compatible transformer so the
exact same fitted object is used at training time and at inference time (no train/serve
skew, no leakage from separately-fit scalers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

PCA_COLS = [f"V{i}" for i in range(1, 29)]


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds amount_log1p, amount_zscore (scaler fit on train), hour_of_day.
    Passes through V1..V28 unchanged (already PCA-transformed upstream).
    Drops raw Time (see configs/features.yaml for rationale).
    """

    def __init__(self):
        self.amount_scaler = StandardScaler()
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        amount_log1p = np.log1p(X["Amount"].clip(lower=0)).values.reshape(-1, 1)
        self.amount_scaler.fit(amount_log1p)
        self.feature_names_ = PCA_COLS + ["amount_log1p", "amount_zscore", "hour_of_day"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[PCA_COLS].copy()
        amount_log1p = np.log1p(X["Amount"].clip(lower=0))
        out["amount_log1p"] = amount_log1p
        out["amount_zscore"] = self.amount_scaler.transform(
            amount_log1p.values.reshape(-1, 1)
        ).ravel()
        out["hour_of_day"] = ((X["Time"].values % 86400) // 3600).astype(int)
        return out[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
