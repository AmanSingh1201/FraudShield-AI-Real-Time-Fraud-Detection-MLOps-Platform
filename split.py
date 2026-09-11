"""Temporal train/validation/test split.

We use a TEMPORAL split, not a random stratified split, because:
  1. The dataset has a genuine time axis (`Time`, seconds elapsed).
  2. In production, a fraud model is always trained on the past and scored on the
     future. A random split lets the model "see" transactions that happened after
     the ones it will be evaluated on, which overstates real-world performance
     (a form of temporal leakage) and hides degradation the model would actually
     face in deployment (fraud patterns, spending patterns, and PCA-feature
     distributions can drift over time).
  3. A temporal split is the closer approximation to how the model will actually
     be evaluated after deployment (train on window N, validate on N+1, serve on
     N+2), so validation/test performance under this split is a more honest proxy
     for production performance than random stratified k-fold would be.

Duplicates (exact duplicate rows) are dropped from the TRAIN split only, per
configs/data.yaml — val/test are left untouched because production would also
see naturally occurring duplicate-looking transactions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                    time_col: str = "Time", drop_dupes_in_train: bool = True):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if drop_dupes_in_train:
        before = len(train)
        train = train.drop_duplicates().reset_index(drop=True)
        dropped = before - len(train)
    else:
        dropped = 0

    # Leakage guard: assert no temporal overlap
    assert train[time_col].max() <= val[time_col].min(), "Temporal leakage: train overlaps val"
    assert val[time_col].max() <= test[time_col].min(), "Temporal leakage: val overlaps test"

    meta = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_time_range": [float(train[time_col].min()), float(train[time_col].max())],
        "val_time_range": [float(val[time_col].min()), float(val[time_col].max())],
        "test_time_range": [float(test[time_col].min()), float(test[time_col].max())],
        "train_fraud_rate": float(train["Class"].mean()),
        "val_fraud_rate": float(val["Class"].mean()),
        "test_fraud_rate": float(test["Class"].mean()),
        "duplicates_dropped_from_train": dropped,
    }
    return train, val, test, meta


def main(raw_path="data/raw/creditcard.csv", out_dir="data/processed"):
    df = pd.read_csv(raw_path)
    train, val, test, meta = temporal_split(df)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    with open(out / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
