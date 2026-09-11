"""Phase 7 (cont.): final evaluation on the untouched test split.

This is the FIRST and ONLY time the test set is touched in this project.
Threshold was selected entirely on the validation set (training/optimize_threshold.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                              fbeta_score, precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: F401,E402

test_df = pd.read_csv("data/processed/test.csv")
y_test = test_df["Class"].values

feat = joblib.load("data/features/feature_pipeline.joblib")
X_test = feat.transform(test_df).values

model = joblib.load("data/features/models/xgboost_optuna_best.joblib")
model_config = json.load(open("data/features/models/model_config.json"))
threshold = model_config["decision_threshold"]

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# precision@k / recall@k: rank-order top-k transactions by predicted probability
k_values = [10, 25, 50, 100]
precision_at_k = {}
recall_at_k = {}
order = np.argsort(-y_prob)
y_sorted = y_test[order]
total_positives = int(y_test.sum())
for k in k_values:
    top_k = y_sorted[:k]
    precision_at_k[k] = float(top_k.sum() / k)
    recall_at_k[k] = float(top_k.sum() / total_positives) if total_positives else 0.0

result = {
    "split": "test (untouched, first evaluation)",
    "n_rows": int(len(y_test)),
    "n_fraud": int(y_test.sum()),
    "threshold_used": threshold,
    "pr_auc_MEASURED": float(average_precision_score(y_test, y_prob)),
    "roc_auc_MEASURED": float(roc_auc_score(y_test, y_prob)),
    "precision_MEASURED": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall_MEASURED": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_MEASURED": float(f1_score(y_test, y_pred, zero_division=0)),
    "f2_MEASURED": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
    "confusion_matrix_MEASURED": {
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    },
    "fraud_capture_rate_MEASURED": float(tp / (tp + fn)) if (tp + fn) else None,
    "false_positive_rate_MEASURED": float(fp / (fp + tn)) if (fp + tn) else None,
    "precision_at_k_MEASURED": precision_at_k,
    "recall_at_k_MEASURED": recall_at_k,
}

Path("docs").mkdir(exist_ok=True)
with open("docs/final_test_evaluation_MEASURED.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
