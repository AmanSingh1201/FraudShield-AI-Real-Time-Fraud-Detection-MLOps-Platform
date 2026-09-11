# Data Pipeline

## Steps, in order

1. **Download** (`make data`): pulls `creditcard.csv` from the public GitHub
   mirror listed in `data/README.md`. Not committed to git.
2. **Validate** (`src/fraudshield/data/validate.py`): schema check (all
   expected columns present, no unexpected ones), null counts, duplicate
   count, class balance, `Time`/`Amount` sanity ranges, dtype check. Fails
   loudly (raises) if a required column is missing. Real output:
   `data/processed/validation_report.json` — **measured**: 0 nulls, 1,081
   exact-duplicate rows (out of 284,807), fraud rate 0.1727%, `Time` confirmed
   monotonic non-decreasing.
3. **Split** (`src/fraudshield/data/split.py`): temporal split (see rationale
   below), 70/15/15 by row count after sorting on `Time`. Duplicates dropped
   from train only. Asserts no time-range overlap between splits before
   writing output.
4. **Feature engineering** (`src/fraudshield/features/pipeline.py`): a single
   `fit`/`transform` object — `fit()` is called on train only, `transform()`
   is called identically at training and serving time. This is the leakage
   guard: there is no code path where a scaler statistic from val/test data
   can influence train-time features.
5. **Training / tuning / evaluation**: see `docs/experiments.md`.

## Why a temporal split, not random stratified

A random stratified k-fold split would let the model train on transactions
that occurred chronologically *after* transactions in its own test set. In
production the model only ever sees the future — it can never train on data
from after the point it's making a prediction. A temporal split (train = past,
val = near-past, test = most-recent) is the split that actually matches how
the model will be evaluated in deployment, and — importantly for this
project — it surfaced a real, measurable phenomenon: the fraud rate and
several feature distributions **already drift** across the 2-day window
(train fraud rate 0.184% → val 0.131% → test 0.122%; 14/29 features flagged
by PSI train-vs-test, see `docs/drift_report_MEASURED.json`). A random split
would have hidden this by shuffling all three periods together, and the
reported test metrics would have been optimistic versus what a temporally-
faithful evaluation shows (val PR-AUC 0.866 vs. test PR-AUC 0.759 — see
`docs/experiments.md` Experiment 008–009).

## Leakage checks implemented (not just claimed)

| Risk | Check | Where |
|---|---|---|
| Temporal leakage | `assert train.Time.max() <= val.Time.min()` and same for val/test | `split.py` |
| Target leakage | `V1..V28`/`Amount` are provider-computed pre-label; no feature is a function of `Class` | by construction |
| Train/test contamination | Split by monotonic `Time` boundary, not random sampling | `split.py` |
| Duplicate rows | Counted (`validate.py`), dropped from train only | `validate.py`, `split.py` |
| Preprocessing leakage | `StandardScaler` inside `FraudFeatureTransformer` is `fit()` on train only; unit test `test_zscore_fit_on_train_only_not_leaked` asserts transforming other data does not mutate the fitted scaler | `features/pipeline.py`, `tests/unit/test_feature_pipeline.py` |
| Post-transaction / future features | Raw `Time` excluded from the model's feature set (see `configs/features.yaml`); only cyclical `hour_of_day`, known at transaction time, is used | `features/pipeline.py` |

## Known data quality issues (measured)
- 1,081 exact duplicate rows in the raw file (0.38% of rows) — plausible
  given this is an anonymized export, not necessarily a data-entry error;
  handled by dropping from train only (val/test are left as-is, since
  production would also see naturally occurring duplicate-looking events).
- No missing values in any column (measured, `validation_report.json`).
- `Amount` ranges from 0 to 25,691.16 with a strong right skew — motivated
  the `amount_log1p` feature (Experiment 005).
