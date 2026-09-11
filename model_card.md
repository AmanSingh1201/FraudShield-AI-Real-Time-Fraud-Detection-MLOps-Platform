# Model Card — FraudShield XGBoost v1 (`xgb-optuna-v1`)

## Intended use
Real-time scoring of individual card-present/card-not-present transactions to
support a fraud-review workflow: assign a fraud probability, map it to a risk
band (LOW / MEDIUM-REVIEW / HIGH-BLOCK) via a separately configurable policy
layer, and return an explanation of the top contributing factors.

Intended consumers: a transaction-processing pipeline calling `POST /predict`
synchronously before authorization, or a batch review queue calling
`POST /predict/batch`.

## Out-of-scope use
- **Not** a standalone auto-decline system: the HIGH_RISK_BLOCK band is
  designed to route to a review/step-up-auth process, not to unilaterally
  deny a transaction without any human or secondary check, given the model's
  measured 28% precision at the chosen operating point (see Metrics).
- **Not** validated for transaction types, currencies, merchant categories, or
  geographies outside the training distribution (European cardholders,
  September 2013, 2-day window) — see Limitations.
- **Not** a general-purpose anomaly detector; it is trained specifically on
  labeled `Class` fraud, not on arbitrary "unusual" transactions.
- **Not** an identity-verification or KYC tool.

## Training data
- ULB Credit Card Fraud Detection dataset (`data/README.md` for full
  provenance): 284,807 transactions, 492 labeled fraud (0.173%), 2013,
  European cardholders, features `V1`–`V28` (PCA-anonymized), `Time`, `Amount`.
- Temporal split: train = first 198,648 rows by `Time` (70%), val = next
  42,721 (15%), test = final 42,722 (15%). Fraud rate drifts across the split:
  train 0.184%, val 0.131%, test 0.122% (**measured**, `data/processed/split_meta.json`).
  This in itself indicates the fraud generating process is not perfectly
  stationary even within a 2-day window.
- 716 exact-duplicate rows dropped from train only.

## Evaluation data
Final numbers below are from the **held-out test split**, touched exactly
once, after the threshold was already fixed on validation.

## Metrics (test split, `xgb-optuna-v1`, threshold = 0.0423)

| Metric | Value | Source |
|---|---|---|
| PR-AUC | 0.7591 | MEASURED |
| ROC-AUC | 0.9784 | MEASURED |
| Precision | 0.2797 | MEASURED |
| Recall (fraud capture rate) | 0.7692 | MEASURED |
| F1 | 0.4103 | MEASURED |
| F2 | 0.5698 | MEASURED |
| False positive rate | 0.00241 | MEASURED |
| Precision@10 / Recall@10 | 1.00 / 0.192 | MEASURED |
| Precision@50 / Recall@50 | 0.78 / 0.75 | MEASURED |

Confusion matrix (test, n=42,722, 52 actual fraud): TN=42,567, FP=103,
FN=12, TP=40.

Full experiment history and interpretation: `docs/experiments.md`.

## Limitations
1. **Opaque features.** `V1`–`V28` are PCA components computed by the data
   provider before this project ever saw the data. SHAP attributions
   (`docs/shap_global_importance_MEASURED.json`) point at `V14`, `V10`, `V4`,
   `V12`, `V17` etc. — these cannot be translated into a business-readable
   explanation like "new device" or "high-risk merchant category" for this
   dataset. A reviewer using `/explain` sees which anonymized components
   drove a decision, not a plain-English reason.
2. **Small, short window.** Only 2 days of data, 492 positives total, 52 in
   the test set. Test-set metric point estimates have wide implicit
   confidence intervals given n=52 positives; a single-digit change in TP/FN
   moves recall by ~2 points. Do not treat these numbers as more precise than
   that.
3. **Val-to-test generalization gap.** Val PR-AUC 0.866 vs. test PR-AUC
   0.759, and precision at the fixed threshold fell from the target 0.30 (val)
   to 0.28 (test) with recall dropping from 0.89 to 0.77. This is consistent
   with real, measured feature drift between the train and test time windows
   (14/29 features flagged by PSI, `docs/drift_report_MEASURED.json`) — i.e.
   the model is already showing the kind of degradation production monitoring
   is meant to catch, inside the same 2-day dataset.
4. **No entity/velocity features.** No card, device, or merchant ID exists in
   this dataset, so none of the standard fraud-ops behavioral features
   (transaction velocity, "new device for this card," merchant risk score)
   could be engineered. See `data/README.md` for the dataset-selection
   trade-off that caused this.
5. **Single time period, single geography.** No evidence this model
   generalizes to non-European cardholders, other years, other payment
   networks, or transaction types not represented in September 2013 European
   card-present/card-not-present data.

## Known biases
Not formally audited for demographic bias (the dataset has no demographic
fields to audit against — a limitation of the dataset, not evidence of
fairness). Any deployment on a dataset with demographic signal would need a
dedicated fairness evaluation before use; this has not been done here.

## Failure cases (observed / expected from the metrics above)
- 12 of 52 test-set frauds were missed (false negatives) at the chosen
  threshold — these tend to be transactions whose PCA-feature profile more
  closely resembles the legitimate majority class.
- 103 of 42,669 legitimate test transactions were flagged (false positives)
  — expected given the 30%-precision design target; this is a review-queue
  volume decision, not a bug.
- Precision@10 = 1.0 but recall@10 = 0.19: the model ranks its most-confident
  predictions very well, but a large fraction of total fraud value is spread
  across lower-confidence scores — a pure top-k review policy would miss most
  fraud.

## Explainability limitations
SHAP TreeExplainer values are exact for this tree ensemble (no approximation
error from the explainer itself), but they explain the model's *use of its
inputs*, not the ground-truth *cause* of fraud, and — as above — those inputs
are anonymized PCA components, capping how actionable the explanation is for
a human reviewer on this particular dataset.

## Monitoring requirements
- Feature-level PSI (train vs. rolling production window) — already shown to
  fire real WARNING/CRITICAL flags on this data (`docs/drift_report_MEASURED.json`).
- Prediction-probability KS-test drift.
- Fraud-rate-when-labels-arrive tracking (delayed-label problem, see
  `docs/monitoring.md`).
- Retraining trigger + promotion gate implemented in `training/register_model.py`
  (a candidate must beat current Production val PR-AUC to be promoted).
