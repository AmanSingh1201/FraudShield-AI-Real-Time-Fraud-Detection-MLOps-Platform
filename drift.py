"""Phase 21: data drift monitoring.

Method chosen: Population Stability Index (PSI), computed per feature between a
reference window (train) and a comparison window (val, test, or production).
PSI is chosen over KS/JS here because it's the industry-standard metric for
tabular *credit risk / fraud* feature monitoring (interpretable fixed
thresholds, cheap to compute per-feature at scale, works for both numeric-
binned and categorical features) -- KS is better for a single continuous
score's distribution (we also use it for the prediction-probability drift
check below, since that IS a single continuous score), and JS divergence adds
symmetry/boundedness we don't need here. Using both PSI-everywhere and
KS-everywhere would be redundant, so each is applied to what it's best at.

Thresholds (standard industry convention):
  PSI < 0.10            -> no significant shift
  0.10 <= PSI < 0.25     -> moderate shift, WARNING
  PSI >= 0.25            -> major shift, CRITICAL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    comp_pct = np.clip(comp_counts / max(len(comparison), 1), 1e-4, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def classify(psi_value: float) -> str:
    if psi_value >= 0.25:
        return "CRITICAL"
    if psi_value >= 0.10:
        return "WARNING"
    return "OK"


def feature_drift_report(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                          columns: list[str]) -> dict:
    report = {}
    for col in columns:
        val = psi(reference_df[col].values, comparison_df[col].values)
        report[col] = {"psi_MEASURED": round(val, 4), "status": classify(val)}
    return report


def prediction_drift_report(reference_probs: np.ndarray, comparison_probs: np.ndarray) -> dict:
    stat, p_value = ks_2samp(reference_probs, comparison_probs)
    return {
        "ks_statistic_MEASURED": round(float(stat), 4),
        "p_value_MEASURED": round(float(p_value), 6),
        "status": "WARNING" if p_value < 0.01 else "OK",
    }


def inject_synthetic_shift(df: pd.DataFrame, amount_multiplier: float = 3.0) -> pd.DataFrame:
    """Only 2 days of real data exist, so there is no real week-over-week window to
    monitor. To demonstrate the drift-detection code path actually triggers on a
    real shift (not just report all-zeros because nothing changed), we apply a
    clearly-labeled SYNTHETIC perturbation to a copy of the data and show PSI
    correctly flags it. This is explicitly a synthetic demo, not a claim about
    production data."""
    shifted = df.copy()
    shifted["Amount"] = shifted["Amount"] * amount_multiplier
    return shifted


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    real_report = feature_drift_report(train_df, test_df, feature_cols)
    n_warn = sum(1 for v in real_report.values() if v["status"] != "OK")

    synthetic_df = inject_synthetic_shift(test_df)
    synthetic_report = feature_drift_report(train_df, synthetic_df, feature_cols)
    n_warn_synth = sum(1 for v in synthetic_report.values() if v["status"] != "OK")

    out = {
        "real_train_vs_test_drift_MEASURED": real_report,
        "real_features_flagged_MEASURED": n_warn,
        "synthetic_shift_demo_note": "Amount x3 injected on a COPY of the test set to prove the detector fires; not a claim about real drift.",
        "synthetic_shift_drift_MEASURED": synthetic_report,
        "synthetic_features_flagged_MEASURED": n_warn_synth,
    }
    Path("docs").mkdir(exist_ok=True)
    with open("docs/drift_report_MEASURED.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Real train-vs-test: {n_warn}/{len(feature_cols)} features flagged (WARNING/CRITICAL)")
    print(f"Synthetic 3x-amount-shift demo: {n_warn_synth}/{len(feature_cols)} features flagged")
    print(json.dumps({"Amount": real_report["Amount"], "Amount_synthetic": synthetic_report["Amount"]}, indent=2))


if __name__ == "__main__":
    main()
