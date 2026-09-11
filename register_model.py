"""Phase 10 (registry): register the selected model in the MLflow Model Registry and
document the Candidate -> Validation -> Production promotion gate.

Promotion rule (enforced here, not just documented): a candidate is only registered
as a new version AND transitioned if it beats the current Production version's
validation PR-AUC. If there is no Production version yet, the first candidate that
passes the minimum bar becomes Production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraudshield.features.pipeline import FraudFeatureTransformer  # noqa: F401,E402

MODEL_NAME = "fraudshield_xgboost"
MIN_ACCEPTABLE_VAL_PR_AUC = 0.70  # minimum bar documented in configs/training.yaml intent

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def get_current_production_pr_auc() -> float | None:
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("val_pr_auc")


def main():
    best = json.load(open("docs/optuna_best_params_MEASURED.json"))
    candidate_pr_auc = best["best_val_pr_auc_MEASURED"]

    if candidate_pr_auc < MIN_ACCEPTABLE_VAL_PR_AUC:
        print(f"REJECTED: candidate val PR-AUC {candidate_pr_auc:.4f} < minimum bar "
              f"{MIN_ACCEPTABLE_VAL_PR_AUC}. Not registering.")
        return

    current_prod_pr_auc = get_current_production_pr_auc()

    mlflow.set_experiment("fraudshield/optuna")
    runs = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("fraudshield/optuna").experiment_id],
        filter_string="attributes.run_name = 'xgboost_optuna_best_FINAL'",
        order_by=["start_time DESC"], max_results=1,
    )
    if not runs:
        print("No FINAL run found to register. Run training/optimize.py first.")
        return
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {result.version} (run_id={run_id})")

    if current_prod_pr_auc is None:
        client.transition_model_version_stage(MODEL_NAME, result.version, "Production")
        print(f"No prior Production model -> promoted v{result.version} directly to Production "
              f"(val PR-AUC {candidate_pr_auc:.4f}).")
    elif candidate_pr_auc > current_prod_pr_auc:
        client.transition_model_version_stage(MODEL_NAME, result.version, "Production",
                                               archive_existing_versions=True)
        print(f"Candidate ({candidate_pr_auc:.4f}) beat current Production "
              f"({current_prod_pr_auc:.4f}) -> promoted.")
    else:
        client.transition_model_version_stage(MODEL_NAME, result.version, "Staging")
        print(f"Candidate ({candidate_pr_auc:.4f}) did NOT beat current Production "
              f"({current_prod_pr_auc:.4f}) -> left in Staging, Production unchanged.")


if __name__ == "__main__":
    main()
