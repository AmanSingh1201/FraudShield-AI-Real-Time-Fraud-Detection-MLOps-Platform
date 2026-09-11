"""Phase 6: Bayesian (TPE) hyperparameter search for XGBoost via Optuna.

Search budget is fixed and controlled (N_TRIALS below) — not an unbounded search.
Objective: maximize PR-AUC on the validation split (never the test split).
Every trial is logged to MLflow under experiment 'fraudshield/optuna'.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: E402

RANDOM_SEED = 42
N_TRIALS = 25  # controlled budget, not unbounded search

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraudshield/optuna")

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
y_train, y_val = train_df["Class"].values, val_df["Class"].values

feat = joblib.load("data/features/feature_pipeline.joblib")
X_train, X_val = feat.transform(train_df).values, feat.transform(val_df).values
scale_pos_weight_base = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", scale_pos_weight_base * 0.3, scale_pos_weight_base * 1.5
        ),
    }
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=False):
        mlflow.log_params(params)
        model = XGBClassifier(
            **params, eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=-1
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        y_prob = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_prob)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("train_time_s", train_time)
    return pr_auc


def main():
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=RANDOM_SEED),
        study_name="fraudshield_xgb_optuna"
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best PR-AUC (val, MEASURED): {study.best_value:.4f}")
    print("Best params:", study.best_params)

    best_params = study.best_params
    final_model = XGBClassifier(
        **best_params, eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    out_dir = Path("data/features/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_dir / "xgboost_optuna_best.joblib")

    import json
    with open("docs/optuna_best_params_MEASURED.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_val_pr_auc_MEASURED": study.best_value,
            "best_params": best_params,
            "n_trials_run": N_TRIALS,
        }, f, indent=2)

    with mlflow.start_run(run_name="xgboost_optuna_best_FINAL"):
        mlflow.log_params(best_params)
        mlflow.log_metric("val_pr_auc", study.best_value)
        mlflow.log_artifact(str(out_dir / "xgboost_optuna_best.joblib"))
        mlflow.xgboost.log_model(final_model, name="model")


if __name__ == "__main__":
    main()
