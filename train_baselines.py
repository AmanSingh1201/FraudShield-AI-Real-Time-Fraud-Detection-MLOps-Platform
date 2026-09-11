"""Phase 5: baseline models, benchmarked under one protocol, tracked in MLflow.

Models: Logistic Regression, Random Forest, XGBoost (all class-weighted).
All are evaluated on the SAME validation split with the SAME metric set.
No metric here is hand-typed; every number comes from sklearn/xgboost computed
on data/processed/{train,val}.csv produced by src/fraudshield/data/split.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score,
                              precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: E402

RANDOM_SEED = 42
DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("data/features")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraudshield/baseline")


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    y = df["Class"].values
    return df, y


def measure_latency(model, X: np.ndarray, n_repeats: int = 500) -> dict:
    """Single-row inference latency, measured (not estimated)."""
    row = X[:1]
    # warm-up
    for _ in range(10):
        model.predict_proba(row)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "latency_ms_mean": float(times.mean()),
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "latency_ms_p99": float(np.percentile(times, 99)),
    }


def evaluate(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "threshold_used": threshold,
        "n_positives_true": int(y_true.sum()),
        "n_positives_pred": int(y_pred.sum()),
    }


def model_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def main():
    train_df, y_train = load_split("train")
    val_df, y_val = load_split("val")

    feat = FraudFeatureTransformer().fit(train_df)
    X_train = feat.transform(train_df).values
    X_val = feat.transform(val_df).values
    joblib.dump(feat, ARTIFACT_DIR / "feature_pipeline.joblib")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_SEED
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results = []
    Path("data/features/models").mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("scale_pos_weight_or_balanced", "balanced/{:.1f}".format(scale_pos_weight))
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("random_seed", RANDOM_SEED)

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_s = time.perf_counter() - t0

            y_prob = model.predict_proba(X_val)[:, 1]
            metrics = evaluate(y_val, y_prob, threshold=0.5)
            latency = measure_latency(model, X_val)

            model_path = Path(f"data/features/models/{name}.joblib")
            joblib.dump(model, model_path)
            size_kb = model_size_kb(model_path)

            row = {
                "model": name,
                "train_time_s_MEASURED": round(train_time_s, 3),
                "model_size_kb_MEASURED": round(size_kb, 1),
                **{f"{k}_MEASURED" if not k.startswith("n_") and not k.startswith("threshold") else k: v
                   for k, v in metrics.items()},
                **{f"{k}_MEASURED": v for k, v in latency.items()},
            }
            results.append(row)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for k, v in latency.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("train_time_s", train_time_s)
            mlflow.log_metric("model_size_kb", size_kb)
            mlflow.log_artifact(str(model_path))

            print(f"[{name}] trained in {train_time_s:.2f}s | "
                  f"PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
                  f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                  f"| p50 latency={latency['latency_ms_p50']:.3f}ms")

    results_df = pd.DataFrame(results)
    out_path = Path("docs/baseline_comparison_MEASURED.csv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(out_path, index=False)
    print("\n=== Baseline comparison (all MEASURED, threshold=0.5) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
