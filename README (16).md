# Dataset

## Candidates evaluated

| Candidate | Scale | Fraud rate | Features | Temporal? | Accessible in this environment? |
|---|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** | ~590K train tx | ~3.5% | 400+ raw (device, card, email, C/D/M/V engineered) | Yes (`TransactionDT` offset) | **No** — Kaggle-hosted, requires Kaggle API auth; Kaggle is not on this environment's network allow-list |
| **ULB / Kaggle "Credit Card Fraud Detection" (2013 European cardholders)** | 284,807 tx | 0.172% (492 fraud) | `Time`, `Amount`, `V1..V28` (PCA-transformed) | Yes (`Time` = seconds elapsed) | **Yes** — mirrored as a CSV on GitHub (`raw.githubusercontent.com` is allow-listed), downloaded directly |
| **PaySim (synthetic mobile money, ~6.3M tx)** | 6.3M | ~0.1% | Sender/receiver balances, transaction type | Yes (`step` = hour) | No public GitHub CSV mirror of usable size found without external hosting (Kaggle-only) |

## Decision

**Selected: ULB Credit Card Fraud Detection dataset** (`creditcard.csv`, 284,807 transactions, 492 frauds).

This is a deliberate, explained trade-off, not a default:

- IEEE-CIS is the stronger dataset for a portfolio project (richer categorical/behavioral feature space, larger scale, closer to real card-network data) and is named as the "strong candidate" in the project brief. However, downloading it requires the Kaggle API with an authenticated `kaggle.json` token, and this build environment's network egress is restricted to a fixed allow-list (PyPI, npm, GitHub, crates.io, Ubuntu archives) that does **not** include `kaggle.com` or Kaggle's CDN. I will not fabricate results against a dataset I cannot actually load.
- The ULB dataset is legitimately hosted as a raw CSV in a public GitHub repo (`raw.githubusercontent.com`, which **is** on the allow-list), so it can be downloaded and used to produce genuinely measured results.
- It is real, is a canonical, non-toy fraud benchmark (used in dozens of published papers), has genuine extreme class imbalance (0.172%), and has a real temporal axis (`Time`), so every requirement in the brief (imbalance handling, temporal split reasoning, threshold optimization, drift monitoring, etc.) can be demonstrated honestly on it.
- **Trade-off accepted:** because `V1..V28` are already PCA components of the original raw features (anonymized by the data provider for privacy), we cannot build interpretable behavioral/velocity features (no card ID, merchant ID, or device ID is available in this dataset — every row is already a single fully-anonymized event). This limits Phase 7 (feature engineering) to `Time`/`Amount`-derived features and PCA-space engineering. This limitation is documented explicitly rather than hidden, and is called out again in `docs/model_card.md` under "Limitations."

If Kaggle access becomes available in this environment (e.g. via an uploaded `kaggle.json` and a network exception), the pipeline in `src/fraudshield/data/` is written to be dataset-agnostic at the loader boundary, and swapping in IEEE-CIS would mainly require a new `configs/data.yaml` + a new feature registry — not a rewrite.

## Source & license

- Original source: Worldline / ULB Machine Learning Group, via Kaggle: `mlg-ulb/creditcardfraud`.
- Also archived on Zenodo (open, versioned): https://zenodo.org/records/7395559
- License: Open / DbCL-style (Kaggle "Open" license on the Zenodo mirror). Not redistributed in this repo — see `.gitignore`; only the download step and derived, non-reversible artifacts (trained models, aggregate stats) are checked in.
- Downloaded for this build from a public GitHub mirror: `nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection` (`raw.githubusercontent.com`), MD5-consistent with the Zenodo copy (150.8MB, 284,807 rows).

## Download instructions (reproduce)

```bash
mkdir -p data/raw
curl -sL -o data/raw/creditcard.csv \
  https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
```
Do not commit `data/raw/creditcard.csv` — it's in `.gitignore`.

## Schema

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed between this transaction and the first transaction in the dataset (covers ~2 days) |
| `V1`–`V28` | float | Principal components from a PCA transform of the original (undisclosed) features, for confidentiality |
| `Amount` | float | Transaction amount |
| `Class` | int {0,1} | Target. 1 = fraud. **492 positives / 284,807 rows = 0.1727% fraud rate** |

## Feature groups

- **Raw anonymized signal**: `V1`–`V28` (already numeric, already scaled by the PCA transform upstream — do not re-standardize blindly, see feature registry).
- **Amount**: raw + engineered (`log1p(Amount)`, amount bucket).
- **Temporal**: `Time` → `hour_of_day` (Time mod 86400 / 3600), `time_since_start`, and rolling second-derived position used only for the split, not as a leaking feature (see below).

## Known limitations

1. No card/user/device/merchant identifiers → no velocity or entity-aggregation features possible (this is the single biggest gap vs. IEEE-CIS / a real production feature store).
2. Only 2 days of data → cannot validate week-over-week or seasonal drift; monitoring demo will use synthetic drift injection for illustration, clearly labeled as such.
3. `V1`–`V28` are opaque PCA components → SHAP explanations will be expressed in terms of `V14`, `V17`, etc., not human-interpretable business features. This is a real limitation of this dataset for the "Explainability" business narrative, and is documented in `docs/model_card.md`.
4. Extremely small positive class (492) means test-set metric confidence intervals are wide — reported with that caveat, not as a settled number.

## Leakage risks investigated

- **Temporal leakage**: `Time` is monotonic in the raw file. We sort by `Time` and use a **temporal split** (train = first ~70% of elapsed time, validation = next ~15%, test = final ~15%) rather than random stratified split, so no transaction "from the future" leaks into training. See `docs/data_pipeline.md`.
- **Target leakage**: none of `V1..V28`/`Amount` are derived from `Class`; they are pre-existing PCA features computed independently by the dataset provider before labeling.
- **Duplicate transactions**: checked programmatically (see `src/fraudshield/data/validate.py`) — duplicates by full feature-row are dropped from train only (kept in test if naturally occurring, since production would see them too).
- **Preprocessing leakage**: any scaler (e.g. for `Amount`) is `fit` on train only and applied to val/test via `transform`, enforced by a single `sklearn.Pipeline` object serialized with the model (see `src/fraudshield/features/pipeline.py`).
