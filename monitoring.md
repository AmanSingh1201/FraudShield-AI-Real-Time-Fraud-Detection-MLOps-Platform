# Monitoring

## What is monitored

| Signal | Method | Implementation | Status here |
|---|---|---|---|
| Feature distribution drift | PSI per feature, train (reference) vs. production window (comparison) | `src/fraudshield/monitoring/drift.py` | Implemented, real numbers on train-vs-test (14/29 flagged) |
| Prediction-probability drift | KS 2-sample test, reference vs. comparison probability distributions | `src/fraudshield/monitoring/drift.py::prediction_drift_report` | Implemented |
| Missingness | Null counts per column | `src/fraudshield/data/validate.py` (offline); same check applicable to a production batch | Implemented offline |
| Categorical distribution changes | N/A for this dataset — no categorical columns exist (`V1..V28`/`Amount`/`Time` are all continuous) | — | Not applicable to this dataset; would use PSI on category frequency for a dataset with categoricals |
| Request volume, decision mix, error rate | In-memory counters | `GET /metrics` in `src/fraudshield/api/main.py` | Implemented (demo-grade: in-process counters, reset on restart — a real deployment would export to Prometheus/StatsD) |
| Latency | Per-request timing in the middleware + structured log line | `src/fraudshield/api/main.py` | Implemented, real measured p50/p95/p99 in `docs/api_latency_benchmark_MEASURED.json` |
| Model performance once labels arrive | See "Delayed labels" below | Design only — no delayed-label feed exists in this offline dataset | Documented, not implemented (see limitation) |

## Why PSI (features) + KS (predictions), not every method

PSI is the standard for per-feature tabular monitoring in credit-risk/fraud
contexts: cheap to compute, interpretable fixed thresholds (0.10/0.25), and
works uniformly whether a feature is binned-continuous or categorical. KS is
applied specifically to the **prediction probability**, a single continuous
score, where a two-sample distributional test is a more natural fit than
binning into PSI buckets. Running both PSI and KS on every signal would be
redundant without adding decision-relevant information, so each method is
scoped to what it does best.

## Thresholds
- PSI < 0.10 → OK. 0.10–0.25 → WARNING. ≥ 0.25 → CRITICAL.
- KS test p-value < 0.01 → WARNING (chosen as a conservative false-positive
  rate for a monitoring alert, not a scientific claim about effect size).
These are configurable constants in `src/fraudshield/monitoring/drift.py`,
not hardcoded inline in a report generator, so ops can tune sensitivity
without touching the computation logic.

## Real measured result on this dataset
Train-vs-test PSI already flags 14 of 29 features as WARNING/CRITICAL, inside
a single 2-day dataset. A synthetic 3x-Amount-shift injected on a copy of the
test set correctly escalates `Amount`'s PSI from 0.023 (OK) to 0.291
(CRITICAL) — confirming the detector fires when a real shift is present, not
just showing zeros because nothing was tested. Full output:
`docs/drift_report_MEASURED.json`.

## Delayed fraud labels

Real fraud labels are not available at prediction time — a customer or issuer
typically confirms fraud days to weeks later via chargeback/dispute. This
means:
- **Cannot** compute live precision/recall in real time; the model's
  precision/recall as reported here are necessarily backward-looking
  (computed against the labeled historical dataset), not a live production
  metric.
- **Design for this project** (not implemented, since there is no live label
  feed in an offline Kaggle-style dataset): maintain a rolling
  "predictions-awaiting-label" table keyed by `transaction_id`; when a label
  arrives (chargeback confirmed / transaction aged out with no dispute =
  implicit negative), join it back to the stored prediction and recompute
  precision/recall/PR-AUC over a trailing window (e.g. last 30 days of
  *labeled* transactions, not last 30 days of *predictions*, since the two
  populations are offset by the label-arrival lag).
- In the interim (before labels arrive), feature-distribution drift (PSI) and
  prediction-distribution drift (KS) are the only real-time signals available
  — this is precisely why both are implemented here even though performance-
  drift monitoring against ground truth isn't.

## Retraining trigger (design, tied to the code that exists)
- **Trigger:** either (a) any feature PSI ≥ 0.25 sustained across N
  consecutive monitoring windows, or (b) trailing labeled-window PR-AUC drops
  below the `MIN_ACCEPTABLE_VAL_PR_AUC` bar already enforced in
  `training/register_model.py` (currently 0.70).
- **Data window:** most recent N days of labeled data plus the existing
  training window (not a full cold-start retrain each time).
- **Validation gate:** re-run `training/optimize_threshold.py` and
  `training/evaluate.py` against a fresh held-out slice; a retrained
  candidate is only promoted if `training/register_model.py`'s comparison
  against current Production val PR-AUC passes — this logic already exists
  and is not new for retraining, it's the same gate used for the initial
  promotion.
- **Rollback:** MLflow Model Registry retains prior Production versions
  (archived, not deleted, by `client.transition_model_version_stage(...,
  archive_existing_versions=True)`); rollback = re-promote the archived
  version, no retraining required.
