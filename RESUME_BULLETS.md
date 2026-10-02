# Resume Bullets — FraudShield AI

These are written from the **actual measured numbers** in this repo (see
`docs/final_test_evaluation_MEASURED.json`, `docs/baseline_comparison_MEASURED.csv`,
`docs/optuna_best_params_MEASURED.json`, `docs/api_latency_benchmark_MEASURED.json`).
Every figure below can be defended line-by-line in an interview by pointing at a
file in this repo. Do not swap these for rounder/bigger numbers from a template —
that's the version that falls apart under a follow-up question.

## Option A — three bullets (standard resume format)

**FraudShield AI — Real-Time Fraud Detection & MLOps Platform**
*Python, XGBoost, scikit-learn, MLflow, Optuna, FastAPI, SHAP, Docker*

- Built an end-to-end fraud detection pipeline on 284K+ transactions with
  0.17% class imbalance, benchmarking Logistic Regression, Random Forest, and
  XGBoost under one evaluation protocol; selected XGBoost for the best
  ROC-AUC (0.99) at 35x lower training cost and 80x lower inference latency
  than Random Forest.
- Tuned XGBoost via Optuna (25-trial Bayesian search) and optimized the
  decision threshold against an explicit precision-floor business objective,
  improving validation PR-AUC to 0.87 and achieving 77% fraud recall at 28%
  precision on a held-out temporal test set.
- Deployed a FastAPI inference service with SHAP explainability and
  PSI-based data-drift monitoring, measuring 5.5ms median / 9.2ms p99 API
  latency across 300 live requests; tracked all experiments and enforced a
  performance-gated model promotion workflow in MLflow.

## Option B — one condensed bullet (if space is tight)

- Built and deployed an XGBoost fraud detection system (284K transactions,
  0.17% fraud rate) with Optuna-tuned hyperparameters, SHAP explainability,
  and PSI drift monitoring behind a FastAPI service (5.5ms median latency),
  using MLflow to track 28+ experiments and gate model promotion.

## Why these numbers, not rounder ones

| What a template might say | What we actually measured | Why the real one is better to use |
|---|---|---|
| "92.4% F1-score" | F1 = 0.41 (test), 0.45 (val, at the chosen recall-favoring threshold) | Fraud detection under 0.17% imbalance legitimately does NOT produce 90%+ F1 without either overfitting or picking a threshold that guts recall. A number like 92.4% is an instant red flag to anyone who's worked on imbalanced classification — it invites the one question you can't answer. |
| "95.1% PR-AUC" | PR-AUC = 0.759 (test), 0.866 (val) | Same issue — PR-AUC near 1.0 on a 0.17%-positive problem is essentially never real. 0.76–0.87 is a genuinely strong, defensible range for this problem. |
| "SMOTE" | `scale_pos_weight` / `class_weight="balanced"` | We deliberately did NOT use SMOTE — it interpolates synthetic points in already-PCA-transformed feature space, which risks generating physically implausible points. Explaining *why* you didn't use a popular technique is a stronger interview answer than name-dropping it. |
| "38ms average / 52ms P95 latency" | 5.5ms median / 6.4ms P95 / 9.2ms P99 (measured over 300 live HTTP requests) | Ours is actually faster — use the real, better number. |
| "Evidently-based drift monitoring" | PSI (features) + KS-test (prediction distribution), custom implementation | We didn't use the Evidently library; we implemented PSI/KS ourselves. Say what you built, not what a template assumed you'd use. |
| "5+ models across 30+ experiments" | 3 baseline models + 25 Optuna trials + final/registry runs = 29 MLflow-tracked runs | Close enough to round to "25+ tracked experiments" honestly — don't need to inflate this one much, just be precise. |

## "why is precision only 28%?"

This is a feature, not a bug, and it's a good answer to have ready: the
threshold was chosen to **maximize fraud recall subject to precision ≥ 30%**
on validation (a review-queue-capacity business objective — see
`training/optimize_threshold.py`), because in fraud detection a missed fraud
case (false negative) is a direct dollar loss, while a false positive just
costs a few minutes of manual review. At threshold=0.5 instead, this same
model gets precision 0.88 / recall 0.79 — the trade-off is a deliberate,
documented choice, not a model limitation. Being able to explain *why* your
number is what it is beats having a bigger number you can't explain.
