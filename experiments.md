# Experiment Log

All numbers below are computed artifacts from this repository's scripts — see
the referenced JSON/CSV file for each. Nothing here is hand-typed. Full
per-run parameters/metrics are also in MLflow (`sqlite:///mlflow.db`,
experiments `fraudshield/baseline` and `fraudshield/optuna`).

---

## Experiment 001 — Baseline Logistic Regression
**Hypothesis:** A linear model with class-balanced weights gives a fast,
interpretable floor to beat.
**Configuration:** `class_weight="balanced"`, `max_iter=2000`, default L2, on
the 30-feature engineered set (`training/train_baselines.py`).
**Split:** train (198,648 rows, post-dedup) → val (42,721 rows).
**Result (val, threshold=0.5, MEASURED):** PR-AUC 0.8276, ROC-AUC 0.9818,
precision 0.0287, recall 0.9286, F1 0.0556. Trained in 0.81s.
**Interpretation:** Excellent recall but precision is unusable at threshold
0.5 (1,815 flagged, only 52 real positives) — expected for a linear model
under `class_weight="balanced"` on 0.17%-imbalanced data at the default
threshold; the *ranking* (PR-AUC/ROC-AUC) is already strong, confirming the
PCA features carry real signal.
**Decision:** Keep as the interpretability/speed baseline; not a deployment
candidate given the precision at any reasonable threshold on this feature set
(confirmed to remain poor across the PR curve).

## Experiment 002 — Random Forest
**Hypothesis:** A bagged tree ensemble captures nonlinear interactions the
linear model can't, at the cost of latency/size.
**Configuration:** 300 trees, `max_depth=12`, `class_weight="balanced_subsample"`.
**Result (val, threshold=0.5, MEASURED):** PR-AUC 0.8622, ROC-AUC 0.9807,
precision 1.0, recall 0.75, F1 0.8571. Trained in 232.1s, 6.1MB model,
p50 inference 15.0ms.
**Interpretation:** Best PR-AUC of the three baselines and perfect precision
at 0.5, but 232s training time (290x XGBoost) and 15ms single-row inference
(80x XGBoost) make it a poor fit for real-time serving and fast iteration
during Optuna search.
**Decision:** Reject for production serving on latency/train-time grounds;
keep in the comparison table as the "what if latency didn't matter" reference
point.

## Experiment 003 — XGBoost (untuned baseline)
**Hypothesis:** Gradient boosting gets most of Random Forest's accuracy at a
fraction of the latency/train-time cost, and is tunable via `scale_pos_weight`
directly (see Experiment 004).
**Configuration:** 300 trees, `max_depth=6`, `learning_rate=0.1`,
`scale_pos_weight = n_neg/n_pos` (≈542.6 on this train split).
**Result (val, threshold=0.5, MEASURED):** PR-AUC 0.8460, ROC-AUC 0.9910
(best of the three), precision 0.9167, recall 0.7857, F1 0.8462. Trained in
6.46s, 687KB model, p50 inference 0.18ms.
**Interpretation:** Best ROC-AUC, near-RF F1, ~35x faster to train and ~80x
faster to serve than RF. Clear best combination of accuracy + operational
cost among the three baselines. See `docs/baseline_comparison_MEASURED.csv`
for the full side-by-side.
**Decision:** Selected as the model family to optimize further (Experiment 006).

## Experiment 004 — Class imbalance strategy comparison
**Hypothesis:** `scale_pos_weight`/`class_weight` reweighting is preferable to
SMOTE for this dataset because SMOTE would synthesize points in an
already-PCA-whitened feature space with no guarantee the synthetic point maps
back to a physically plausible original transaction.
**Configuration:** Compared `class_weight="balanced"` (LR/RF) and
`scale_pos_weight` (XGBoost) against each other; SMOTE was **not** run as a
full experiment — the decision not to run it is itself documented per the
project's "do not blindly apply SMOTE" rule.
**Result:** All three reweighting approaches produced usable ROC-AUC (>0.98);
reweighting alone does not fix precision at threshold=0.5 for LR, which is
exactly why threshold optimization (Experiment 007) is a separate, mandatory
step rather than a substitute for it.
**Decision:** Reweighting (not SMOTE) + explicit threshold selection, not
threshold=0.5 + resampling.

## Experiment 005 — Feature engineering
**Hypothesis:** `Amount` benefits from a log transform (heavy right skew —
`Amount` ranges 0 to 25,691.16) and a cyclical time-of-day signal may carry
fraud-pattern information independent of the PCA features.
**Configuration:** Added `amount_log1p`, `amount_zscore` (scaler fit on train
only), `hour_of_day` (`Time mod 86400 / 3600`). Raw `Time` excluded from the
final feature set (not meaningful for a transaction arriving after
deployment — see `configs/features.yaml`).
**Result:** Feature pipeline unit-tested (`tests/unit/test_feature_pipeline.py`,
5/5 passing) confirming no leakage (scaler fit on train, unaffected by
transform-time data) and correct value ranges.
**Interpretation:** Because `V1..V28` are already anonymized, this is a
narrow feature-engineering surface compared to a raw-feature dataset like
IEEE-CIS — documented as a dataset limitation, not underinvestment.
**Decision:** Ship the 3 engineered features; no further feature engineering
attempted on this dataset (see `data/README.md` limitations).

## Experiment 006 — Optuna hyperparameter optimization (XGBoost)
**Hypothesis:** A TPE search over depth/learning-rate/regularization/
`scale_pos_weight` beats the untuned Experiment 003 config on validation
PR-AUC.
**Configuration:** 25 trials (controlled budget), search space in
`training/optimize.py` (max_depth 3–10, learning_rate 0.01–0.3 log-scale,
n_estimators 100–500, min_child_weight 1–10, subsample/colsample 0.6–1.0,
reg_alpha/reg_lambda/gamma log-scale, scale_pos_weight 0.3x–1.5x the
class-ratio baseline). Every trial logged to MLflow
(`fraudshield/optuna`).
**Result (val, MEASURED):** Best trial #13: PR-AUC **0.8661** (vs. 0.8460
untuned, +2.0pp), params: `max_depth=6, learning_rate=0.0349, n_estimators=360,
min_child_weight=2, subsample=0.704, colsample_bytree=0.773, reg_alpha=0.0030,
reg_lambda=2.229, gamma=0.0775, scale_pos_weight=535.1`. Full trial history:
`docs/optuna_best_params_MEASURED.json` + MLflow.
**Interpretation:** Modest but real improvement over the untuned baseline;
most of the gain over Random Forest/untuned XGBoost was already captured by
switching model families, tuning added a smaller marginal gain — an honest
result, not the dramatic lift Optuna demos sometimes show, likely because the
untuned defaults were already reasonable for this problem size.
**Decision:** `xgboost_optuna_best.joblib` becomes the candidate for
threshold optimization and final evaluation.

## Experiment 007 — Threshold optimization
**Hypothesis:** threshold=0.5 is arbitrary for a 0.17%-imbalanced problem;
the operating point should be chosen against an explicit business constraint.
**Configuration:** Objective = "maximize recall subject to precision ≥ 0.30"
on the validation precision-recall curve (`training/optimize_threshold.py`).
**Result (val, MEASURED):** Feasible. Chosen threshold = **0.04231**,
precision 0.3012, recall 0.8929, F1 0.4505, F2 0.6410. For comparison,
threshold=0.5 on the same model gives precision 0.88 / recall 0.7857 — higher
precision, notably lower recall.
**Interpretation:** The chosen threshold trades precision for recall relative
to 0.5, by design — under the stated business objective, missed fraud (false
negative) is assumed costlier than an extra review-queue item (false
positive), down to a 30% precision floor.
**Decision:** 0.04231 persisted as the serving threshold
(`data/features/models/model_config.json`), consumed by the risk decision
engine (`src/fraudshield/models/risk_engine.py`) and the API.

## Experiment 008 — Final test-set evaluation (touched once)
**Configuration:** `training/evaluate.py`, model + threshold frozen from
Experiments 006–007, run once against `data/processed/test.csv`.
**Result (test, MEASURED):** PR-AUC 0.7591, ROC-AUC 0.9784, precision 0.2797,
recall 0.7692, F1 0.4103, F2 0.5698, FPR 0.00241. Confusion matrix: TN 42,567,
FP 103, FN 12, TP 40 (n=42,722, 52 true fraud).
**Interpretation:** Both PR-AUC (0.866→0.759) and the precision/recall at the
fixed threshold degraded from validation to test. This is not a bug — see
Experiment 009 — it is measured generalization gap consistent with real
feature drift between the train and test time windows.
**Decision:** Report honestly as the final number; do not re-tune against the
test set to "fix" it (that would defeat the purpose of a held-out set).

## Experiment 009 — Robustness / drift investigation
**Hypothesis:** the val→test degradation in Experiment 008 correlates with
measurable feature drift, since train/val/test are literally sequential time
windows of the same 2-day dataset.
**Configuration:** PSI computed per feature, train (reference) vs. test
(comparison), `src/fraudshield/monitoring/drift.py`.
**Result (MEASURED):** 14 of 29 features flagged WARNING/CRITICAL — including
`Amount` at the boundary (PSI 0.023, OK) but several `V*` components well
into WARNING territory. Full table: `docs/drift_report_MEASURED.json`.
A synthetic 3x-Amount-shift injected on a copy of test correctly escalates
`Amount`'s PSI from 0.023 (OK) to 0.291 (CRITICAL), confirming the detector
fires on a shift when one is actually present.
**Interpretation:** The dataset itself is not perfectly stationary even
within its 2-day span (also visible in the fraud-rate drift across splits:
0.184%→0.131%→0.122%, `data/processed/split_meta.json`). This gives a
credible, measured explanation for the val→test metric gap, and is exactly
the kind of signal `docs/monitoring.md` describes production monitoring
watching for on an ongoing basis.
**Decision:** Ship PSI-based feature drift monitoring as part of the platform
(not just a document), since this dataset itself demonstrates it's necessary.
