"""Phase 8: SHAP explainability for the deployed XGBoost model.

Limitation (documented, see data/README.md and docs/model_card.md): the PCA
features V1..V28 are already anonymized components from the data provider, so
SHAP attributions point at "V14", "V17", etc. rather than human-interpretable
business features (no "unusual merchant category" or "new device" signal is
available in this dataset). This is a real, disclosed limitation of the chosen
dataset, not a limitation of the SHAP methodology itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / ".."))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: F401,E402


class FraudExplainer:
    def __init__(self, model_path="data/features/models/xgboost_optuna_best.joblib",
                 feature_pipeline_path="data/features/feature_pipeline.joblib"):
        self.model = joblib.load(model_path)
        self.feat = joblib.load(feature_pipeline_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self.feat.feature_names_

    def explain_row(self, raw_row: pd.DataFrame, top_n: int = 5) -> dict:
        X = self.feat.transform(raw_row)
        shap_values = self.explainer.shap_values(X)
        values = shap_values[0] if shap_values.ndim == 2 else shap_values
        contributions = list(zip(self.feature_names, values.tolist(), X.iloc[0].tolist()))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        top = contributions[:top_n]
        return {
            "top_factors": [
                {"feature": f, "shap_value": round(v, 5), "feature_value": round(fv, 5),
                 "direction": "increases_fraud_risk" if v > 0 else "decreases_fraud_risk"}
                for f, v, fv in top
            ]
        }

    def global_importance(self, X_sample: np.ndarray, sample_size: int = 2000) -> dict:
        if len(X_sample) > sample_size:
            idx = np.random.RandomState(42).choice(len(X_sample), sample_size, replace=False)
            X_sample = X_sample[idx]
        shap_values = self.explainer.shap_values(X_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(zip(self.feature_names, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
        return {"global_feature_importance_MEASURED": [
            {"feature": f, "mean_abs_shap": round(v, 6)} for f, v in ranked
        ]}


def main():
    val_df = pd.read_csv("data/processed/val.csv")
    explainer = FraudExplainer()
    X_val = explainer.feat.transform(val_df).values

    global_imp = explainer.global_importance(X_val)
    Path("docs").mkdir(exist_ok=True)
    with open("docs/shap_global_importance_MEASURED.json", "w") as f:
        json.dump(global_imp, f, indent=2)
    print("Top 10 global features by mean |SHAP|:")
    for row in global_imp["global_feature_importance_MEASURED"][:10]:
        print(f"  {row['feature']:>16s}  {row['mean_abs_shap']:.5f}")

    fraud_rows = val_df[val_df["Class"] == 1]
    if len(fraud_rows):
        example = fraud_rows.iloc[[0]]
        local = explainer.explain_row(example)
        with open("docs/shap_local_example_MEASURED.json", "w") as f:
            json.dump(local, f, indent=2)
        print("\nLocal explanation for one true-fraud validation example:")
        print(json.dumps(local, indent=2))


if __name__ == "__main__":
    main()
