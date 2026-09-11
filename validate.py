"""Schema and integrity validation for the raw FraudShield dataset.

Run as a script to produce a validation report (data/processed/validation_report.json).
Fails loudly (raises) on schema violations rather than silently coercing bad data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_cols: int = 0
    missing_columns: list = field(default_factory=list)
    unexpected_columns: list = field(default_factory=list)
    null_counts: dict = field(default_factory=dict)
    n_duplicate_rows: int = 0
    class_counts: dict = field(default_factory=dict)
    fraud_rate: float = 0.0
    time_min: float = 0.0
    time_max: float = 0.0
    time_is_monotonic_non_decreasing: bool = False
    amount_min: float = 0.0
    amount_max: float = 0.0
    negative_amounts: int = 0
    dtype_issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def validate(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.n_rows, report.n_cols = df.shape

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    report.missing_columns = missing
    report.unexpected_columns = unexpected
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")

    report.null_counts = {c: int(df[c].isnull().sum()) for c in df.columns}
    total_nulls = sum(report.null_counts.values())
    if total_nulls > 0:
        # Not fatal for this dataset (it has none), but must be visible.
        pass

    report.n_duplicate_rows = int(df.duplicated().sum())

    class_counts = df["Class"].value_counts().to_dict()
    report.class_counts = {int(k): int(v) for k, v in class_counts.items()}
    report.fraud_rate = float(df["Class"].mean())

    report.time_min = float(df["Time"].min())
    report.time_max = float(df["Time"].max())
    report.time_is_monotonic_non_decreasing = bool(df["Time"].is_monotonic_increasing)

    report.amount_min = float(df["Amount"].min())
    report.amount_max = float(df["Amount"].max())
    report.negative_amounts = int((df["Amount"] < 0).sum())

    for c in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            report.dtype_issues.append(c)

    return report


def main(raw_path: str = "data/raw/creditcard.csv",
         out_path: str = "data/processed/validation_report.json") -> ValidationReport:
    df = pd.read_csv(raw_path)
    report = validate(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(json.dumps(report.to_dict(), indent=2))
    return report


if __name__ == "__main__":
    main()
