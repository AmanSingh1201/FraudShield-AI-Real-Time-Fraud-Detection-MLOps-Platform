# FraudShield AI — Real-Time Fraud Detection & MLOps Platform

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Tests](https://img.shields.io/badge/tests-28%2F28%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end fraud detection system: real dataset → validated, leakage-checked
pipeline → benchmarked models → tuned + threshold-optimized XGBoost →
explainable, policy-driven real-time API → drift monitoring → tests → CI.

**Every metric below is measured by the code in this repo — no fabricated
numbers.** Anything not yet execution-verified (Docker build/run — no Docker
daemon in the build sandbox) is disclosed as such rather than claimed. See
`docs/system_design.md` → "What this design does NOT claim."

## Results at a glance

| Held-out test set (n=42,722, 52 fraud) | Value |
|---|---|
| PR-AUC | **0.7591** |
| ROC-AUC | **0.9784** |
| Precision / Recall | 0.2797 / 0.7692 |
| F1 / F2 | 0.4103 / 0.5698 |
| API latency (p50 / p99, 300 real requests) | 5.46ms / 9.15ms |
| Tests passing | 28 / 28 |

<p align="center">
  <img src="docs/images/model_comparison.png" width="49%">
  <img src="docs/images/precision_recall_curve.png" width="49%">
</p>
<p align="center">
  <img src="docs/images/roc_curve.png" width="49%">
  <img src="docs/images/confusion_matrix.png" width="49%">
</p>
<p align="center">
  <img src="docs/images/shap_global_importance.png" width="49%">
  <img src="docs/images/optuna_search_history.png" width="49%">
</p>
<p align="center">
  <img src="docs/images/api_demo.png" width="49%">
  <img src="docs/images/pytest_results.png" width="49%">
</p>

More: [full experiment log](docs/experiments.md) · [model card](docs/model_card.md) · [resume bullets built from these numbers](docs/RESUME_BULLETS.md)

## 1. Problem statement
Score each transaction for fraud probability in real time, route it to an
APPROVE/REVIEW/BLOCK decision under an explicit, configurable business
objective, and explain why — while handling a target class that is 0.17% of
all transactions.

## 2. Why fraud detection is difficult
Extreme class imbalance (492 fraud / 284,807 transactions), a moving target
(fraud patterns and even the underlying data distribution drift — measured
here, see §14), asymmetric costs (missed fraud vs. false-positive review
cost), and — for this dataset specifically — anonymized features that limit
how interpretable any single prediction can be (§5 limitations).

## 3. Dataset
**ULB Credit Card Fraud Detection** (284,807 transactions, 492 fraud,
0.173%), chosen over IEEE-CIS because IEEE-CIS requires Kaggle API auth not
available in this build environment — full comparison and trade-off analysis
in `data/README.md`. Temporal split (train 70% / val 15% / test 15%, by
`Time`), not random, because production only ever trains on the past — see
`docs/data_pipeline.md`.

## 4. Architecture
See `docs/architecture.md` for the full component diagram (offline
training pipeline, online serving path, monitoring). Summary: one versioned
deployable unit (`feature_pipeline.joblib` + `model.joblib` +
`model_config.json`) served by FastAPI behind nginx, with the risk-decision
policy kept separate from the ML model.

## 5. Feature engineering
`V1`–`V28` pass through unchanged (already PCA-transformed by the data
provider — no card/device/merchant ID exists in this dataset, so no
velocity/entity features were possible; documented trade-off in
`data/README.md`). Engineered: `amount_log1p`, `amount_zscore` (scaler fit on
train only), `hour_of_day`. Full feature registry: `configs/features.yaml`.

## 6. Models evaluated
Logistic Regression, Random Forest, XGBoost — same protocol, same validation
split. Full table with **measured** PR-AUC/ROC-AUC/precision/recall/F1/
train-time/model-size/latency: `docs/baseline_comparison_MEASURED.csv`.
XGBoost selected for the best accuracy/latency/train-time balance (best
ROC-AUC 0.991, 6.5s train time, 0.18ms p50 raw inference, vs. Random Forest's
232s train time / 15ms inference for a similar F1). Reasoning:
`docs/experiments.md`.

## 7. Class imbalance strategy
`class_weight="balanced"` (LR/RF) / `scale_pos_weight` (XGBoost) — not SMOTE,
because SMOTE would interpolate synthetic points in an already-PCA-whitened
feature space with no guarantee of physical plausibility. Full reasoning:
`configs/training.yaml`, `docs/experiments.md` Experiment 004.

## 8. Hyperparameter optimization
Optuna TPE, 25 trials (controlled budget), objective = validation PR-AUC,
every trial logged to MLflow. Best val PR-AUC **0.8661** (vs. 0.8460 untuned
XGBoost, +2.0pp) — an honest, modest improvement, not an inflated demo
number. Full params: `docs/optuna_best_params_MEASURED.json`.

## 9. Threshold optimization
Business objective (explicit, not default 0.5): **maximize recall subject to
precision ≥ 0.30** on validation. Chosen threshold **0.0423** → val precision
0.301 / recall 0.893. Compared against threshold=0.5 (precision 0.88 / recall
0.786) in `docs/threshold_selection_MEASURED.json`.

## 10. Final evaluation (held-out test set, touched once)
| Metric | Value |
|---|---|
| PR-AUC | 0.7591 |
| ROC-AUC | 0.9784 |
| Precision | 0.2797 |
| Recall (fraud capture) | 0.7692 |
| F1 / F2 | 0.4103 / 0.5698 |
| False positive rate | 0.00241 |

Confusion matrix (n=42,722, 52 fraud): TN 42,567 · FP 103 · FN 12 · TP 40.
Full breakdown incl. precision/recall@k: `docs/final_test_evaluation_MEASURED.json`.
**Note the gap vs. validation** (PR-AUC 0.866→0.759) — investigated and
explained in §14, not hidden.

## 11. Explainability
SHAP `TreeExplainer` (exact for tree ensembles), global importance +
per-prediction local explanation, exposed via `POST /explain`. Limitation:
attributions are in terms of anonymized `V*` components, not business-
readable reasons — disclosed in `docs/model_card.md`.

## 12. API
`FastAPI` — `POST /predict`, `POST /predict/batch`, `POST /explain`,
`GET /health`, `GET /model/info`, `GET /metrics`. Pydantic-validated,
structured logging with request IDs, no stack traces ever returned to
clients (tested). Real measured latency (300 live HTTP requests):
p50 **5.46ms**, p95 6.43ms, p99 9.15ms
(`docs/api_latency_benchmark_MEASURED.json`).

## 13. MLOps
MLflow (`sqlite:///mlflow.db`) tracks every baseline + Optuna trial. Model
Registry with a real promotion gate (`training/register_model.py`): a
candidate is only promoted to Production if it beats the current Production
model's validation PR-AUC; prior versions are archived, not deleted, so
rollback is a stage transition, not a retrain.

## 14. Monitoring
PSI (features) + KS (prediction distribution). **Real finding on this
dataset**: 14 of 29 features already show train-vs-test drift
(`docs/drift_report_MEASURED.json`) — which is the most likely explanation
for the val→test metric gap in §10. A synthetic 3x-Amount shift correctly
escalates PSI from 0.023 (OK) to 0.291 (CRITICAL), confirming the detector
actually fires. Delayed-label handling and retraining triggers designed in
`docs/monitoring.md`.

## 15. Deployment
Multi-stage, non-root, healthchecked `deployment/Dockerfile`;
`docker-compose.yml` runs API + MLflow + nginx. **Disclosed limitation:** no
Docker daemon was available in the build sandbox, so the image build itself
is not execution-verified here — written to the same standard as everything
else, but call this out if asked in an interview rather than claiming it was
tested.

## 16. Testing
28 tests, unit + integration + API, **all passing** (`make test`):
feature-pipeline leakage guards, risk-engine boundary logic, PSI correctness,
full inference pipeline against real test-split rows, and FastAPI endpoint
tests (valid/invalid/malformed input, batch, metrics, no-stack-trace-leak).

## 17. Reproduction instructions
```bash
git clone <this-repo> && cd fraudshield-ai
make install
make data          # downloads creditcard.csv from the GitHub mirror in data/README.md
make validate
make split
make baselines     # trains LR/RF/XGBoost, logs to MLflow
make optimize      # Optuna search, 25 trials
make threshold     # threshold optimization on val
make evaluate      # ONE-TIME test-set evaluation
make shap
make drift
make test
make api           # http://localhost:8000/docs
```
Or `make all` to run the full pipeline in order. `make mlflow-ui` to browse
experiments at `http://localhost:5001`.

## 18. Limitations
- Anonymized PCA features → no interpretable business features, no
  velocity/entity aggregation possible (biggest gap vs. IEEE-CIS).
- Only 2 days of data → small positive test set (52 frauds); metrics have
  wide implicit confidence intervals; no real seasonal/weekly validation.
- No live delayed-label feed (offline dataset) — retraining/rollback are
  designed and gate-enforced in code, but not exercised against live traffic.
- Docker build not execution-verified in this environment (no daemon
  available) — see §15.
- No demographic fields exist to audit for fairness — absence of evidence is
  not evidence of fairness; not claimed here.

## What to push to GitHub
This repo is already structured so `git add . && git commit && git push`
does the right thing — `.gitignore` excludes what shouldn't go up:
- **Excluded (correctly):** `data/raw/creditcard.csv` (large + not yours to
  redistribute — link to `data/README.md`'s download command instead),
  `data/processed/*.csv` (regenerable via `make split`), `*.joblib` model
  files (regenerable via `make baselines optimize`, and large — GitHub will
  warn above 50MB), `mlruns/`, `mlflow.db`, `__pycache__/`, `.venv/`.
- **Included (push these):** everything else — `src/`, `training/`, `tests/`,
  `configs/`, `docs/` (including `docs/images/*.png` — small, and this is
  exactly what makes the README render nicely on GitHub), `deployment/`,
  `.github/workflows/ci.yml`, `README.md`, `requirements.txt`, `Makefile`,
  `LICENSE`, `pyproject.toml`, `docker-compose.yml`.

If you do want the trained model artifacts on GitHub for a portfolio repo
(so someone can clone-and-run without retraining first), the honest way is
[Git LFS](https://git-lfs.github.com/) for the `.joblib` files — plain `git
add` on files that size will bloat repo history permanently. Simplest
alternative: leave them out and let `make baselines optimize threshold`
regenerate them in ~2 minutes on clone.


## 19. Future improvements
- Swap in IEEE-CIS (or a real production feature store) for genuine
  velocity/entity features once Kaggle access or an equivalent raw-feature
  dataset is available — the pipeline's loader boundary is already dataset-
  agnostic (`configs/data.yaml`).
- Cost-based threshold objective (expected $ loss) as an alternative to the
  precision-floor objective used here, with a real per-transaction cost model.
- Prometheus/Grafana export for `/metrics` and the drift reports instead of
  in-process counters and JSON files.
- Execution-verify the Docker build/run and add a container-level smoke test
  to CI once a Docker-enabled runner is available.
