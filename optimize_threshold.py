"""Phase 7: threshold optimization.

Business objective chosen for this project (documented, not arbitrary):
  "Maximize fraud recall subject to precision >= 0.30 on the validation set."

Rationale: card-issuer fraud review teams have finite manual-review capacity.
A precision floor of 30% means at most ~7 out of 10 flagged transactions are
false alarms sent to a human reviewer or a step-up auth challenge — a common
real-world constraint range for review-queue-based fraud ops (vs. auto-decline,
which would demand much higher precision). Recall is maximized subject to that
floor because missed fraud is a direct dollar loss, while false positives cost
review time, not principal.

This constraint is a design choice, stated explicitly, and can be swapped for
a cost-based objective (docs/experiments.md discusses the alternative).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, fbeta_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: F401,E402

PRECISION_FLOOR = 0.30

val_df = pd.read_csv("data/processed/val.csv")
y_val = val_df["Class"].values
feat = joblib.load("data/features/feature_pipeline.joblib")
X_val = feat.transform(val_df).values

model = joblib.load("data/features/models/xgboost_optuna_best.joblib")
y_prob = model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
# precision_recall_curve returns len(thresholds) = len(precisions) - 1
precisions_t = precisions[:-1]
recalls_t = recalls[:-1]

feasible = precisions_t >= PRECISION_FLOOR
if feasible.any():
    feasible_recalls = np.where(feasible, recalls_t, -1)
    best_idx = int(np.argmax(feasible_recalls))
    chosen_threshold = float(thresholds[best_idx])
    chosen_precision = float(precisions_t[best_idx])
    chosen_recall = float(recalls_t[best_idx])
    feasible_found = True
else:
    # No threshold meets the floor: fall back to max-F1 and report that the
    # business objective was infeasible on this validation split.
    best_idx = int(np.argmax([f1_score(y_val, (y_prob >= t).astype(int)) for t in thresholds]))
    chosen_threshold = float(thresholds[best_idx])
    chosen_precision = float(precisions_t[best_idx])
    chosen_recall = float(recalls_t[best_idx])
    feasible_found = False

y_pred_chosen = (y_prob >= chosen_threshold).astype(int)
f1_at_chosen = f1_score(y_val, y_pred_chosen)
f2_at_chosen = fbeta_score(y_val, y_pred_chosen, beta=2)

# also report the plain default for comparison
y_pred_05 = (y_prob >= 0.5).astype(int)
from sklearn.metrics import precision_score, recall_score
precision_05 = precision_score(y_val, y_pred_05, zero_division=0)
recall_05 = recall_score(y_val, y_pred_05, zero_division=0)

result = {
    "business_objective": f"maximize recall subject to precision >= {PRECISION_FLOOR}",
    "objective_feasible_on_val_MEASURED": feasible_found,
    "chosen_threshold_MEASURED": round(chosen_threshold, 6),
    "precision_at_chosen_threshold_MEASURED": round(chosen_precision, 4),
    "recall_at_chosen_threshold_MEASURED": round(chosen_recall, 4),
    "f1_at_chosen_threshold_MEASURED": round(float(f1_at_chosen), 4),
    "f2_at_chosen_threshold_MEASURED": round(float(f2_at_chosen), 4),
    "comparison_default_threshold_0.5": {
        "precision_MEASURED": round(float(precision_05), 4),
        "recall_MEASURED": round(float(recall_05), 4),
    },
    "n_val_positives": int(y_val.sum()),
    "n_val_rows": int(len(y_val)),
}

Path("docs").mkdir(exist_ok=True)
with open("docs/threshold_selection_MEASURED.json", "w") as f:
    json.dump(result, f, indent=2)

# Persist threshold alongside model config for the serving pipeline
with open("data/features/models/model_config.json", "w") as f:
    json.dump({
        "model_file": "xgboost_optuna_best.joblib",
        "feature_pipeline_file": "feature_pipeline.joblib",
        "decision_threshold": chosen_threshold,
        "review_threshold": max(chosen_threshold * 0.4, 0.01),  # lower band -> REVIEW zone
        "model_version": "xgb-optuna-v1",
    }, f, indent=2)

print(json.dumps(result, indent=2))
